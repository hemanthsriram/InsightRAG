import os
import time
import tempfile
from dataclasses import dataclass
from typing import Iterable

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


APP_TITLE = "PDF Chat"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE = 10
MAX_RETRIES = 5
RESULT_COUNT = 4


@dataclass(frozen=True)
class ProcessedPdf:
    file_name: str
    chunk_count: int
    page_count: int
    vector_store: FAISS


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="PDF",
        layout="wide",
        initial_sidebar_state="collapsed",
    )


def get_google_api_key() -> str:
    env_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if env_key:
        return env_key

    try:
        return str(st.secrets.get("GOOGLE_API_KEY", "")).strip()
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def get_embeddings(api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key,
    )


@st.cache_resource(show_spinner=False)
def get_llm(api_key: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.1,
        google_api_key=api_key,
    )


def save_upload_to_temp(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return temp_file.name


def load_pdf(file_path: str, file_name: str) -> list[Document]:
    reader = PdfReader(file_path)
    documents: list[Document] = []

    for index, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": index, "source": file_name},
                )
            )

    if not documents:
        raise ValueError("No selectable text was found in this PDF.")

    return documents


def split_documents(documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def _embed_with_retry(
    texts: list[str],
    embeddings: GoogleGenerativeAIEmbeddings,
) -> list[list[float]]:
    """Embed a list of texts with exponential backoff on rate-limit errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        "Embedding failed after multiple retries. "
        "Please try again in a few minutes or use a smaller PDF."
    )


def create_vector_store(chunks: list[Document], api_key: str) -> FAISS:
    """Create a FAISS vector store from document chunks.

    FAISS stores its index as a pure Python object — each call produces
    a completely independent instance with zero shared global state.
    This guarantees that different Streamlit sessions never interfere
    with each other.
    """
    embeddings = get_embeddings(api_key)

    # Embed the first batch and create the FAISS store
    first_batch = chunks[:BATCH_SIZE]
    remaining = chunks[BATCH_SIZE:]

    first_texts = [doc.page_content for doc in first_batch]
    first_vectors = _embed_with_retry(first_texts, embeddings)
    text_embedding_pairs = list(zip(first_texts, first_vectors))

    vector_store = FAISS.from_embeddings(
        text_embeddings=text_embedding_pairs,
        embedding=embeddings,
        metadatas=[doc.metadata for doc in first_batch],
    )

    # Add remaining chunks in batches with retry logic
    for i in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[i : i + BATCH_SIZE]
        batch_texts = [doc.page_content for doc in batch]
        batch_vectors = _embed_with_retry(batch_texts, embeddings)
        batch_pairs = list(zip(batch_texts, batch_vectors))

        batch_store = FAISS.from_embeddings(
            text_embeddings=batch_pairs,
            embedding=embeddings,
            metadatas=[doc.metadata for doc in batch],
        )
        vector_store.merge_from(batch_store)

        # Small delay between batches to stay under rate limits
        time.sleep(1)

    return vector_store


def process_pdf(uploaded_file, api_key: str) -> ProcessedPdf:
    temp_path = save_upload_to_temp(uploaded_file)
    try:
        pages = load_pdf(temp_path, uploaded_file.name)
        chunks = split_documents(pages)
        if not chunks:
            raise ValueError("No text chunks could be created from this PDF.")

        return ProcessedPdf(
            file_name=uploaded_file.name,
            chunk_count=len(chunks),
            page_count=len(pages),
            vector_store=create_vector_store(chunks, api_key),
        )
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def answer_question(
    vector_store: FAISS,
    question: str,
    api_key: str,
) -> tuple[str, list[Document]]:
    docs = vector_store.similarity_search(question, k=RESULT_COUNT)
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question using only the context below. If the answer is not
        present in the context, say that the PDF does not contain enough
        information.

        Context:
        {context}

        Question: {input}
        """
    )
    chain = create_stuff_documents_chain(get_llm(api_key), prompt)
    return chain.invoke({"input": question, "context": docs}), docs


def source_label(document: Document, index: int) -> str:
    page = document.metadata.get("page")
    page_text = f"page {page + 1}" if isinstance(page, int) else "source"
    return f"Source {index}: {page_text}"


def initialize_state() -> None:
    st.session_state.setdefault("processed_pdf", None)
    st.session_state.setdefault("messages", [])


def render_app() -> None:
    st.title(APP_TITLE)

    api_key = get_google_api_key()
    if not api_key:
        st.error("Set GOOGLE_API_KEY before using the app.")
        st.stop()

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    col_a, col_b = st.columns(2)

    with col_a:
        build_clicked = st.button(
            "Start Chat",
            disabled=uploaded_file is None,
            use_container_width=True,
        )
    with col_b:
        clear_clicked = st.button("Clear chat", use_container_width=True)

    if clear_clicked:
        st.session_state.processed_pdf = None
        st.session_state.messages = []
        st.rerun()

    if build_clicked and uploaded_file is not None:
        with st.spinner("Parsing PDF and creating embeddings..."):
            try:
                st.session_state.processed_pdf = process_pdf(uploaded_file, api_key)
                st.session_state.messages = []
                st.success(f"Ready: {uploaded_file.name}")
            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources"):
                    for label, excerpt in message["sources"]:
                        st.caption(label)
                        st.write(excerpt)

    question = st.chat_input("Ask about the PDF")
    if question:
        if not st.session_state.processed_pdf:
            st.warning("Upload a PDF and click 'Start Chat' first.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Answering..."):
                try:
                    answer, docs = answer_question(
                        st.session_state.processed_pdf.vector_store,
                        question,
                        api_key,
                    )
                    sources = [
                        (source_label(document, index), document.page_content[:500])
                        for index, document in enumerate(docs, start=1)
                    ]
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )
                except Exception as exc:
                    st.error(f"Could not generate an answer: {exc}")


def main() -> None:
    configure_page()
    initialize_state()
    render_app()


if __name__ == "__main__":
    main()
