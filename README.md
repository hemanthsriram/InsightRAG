# PDF RAG Studio

A web-based PDF question-answering system rebuilt from
`muqadasejaz/PDF-QA-RAG-System` with a simple Streamlit upload and chat
interface.

The app lets you upload a PDF, build a Chroma vector index, and ask grounded
questions answered by Gemini through the Google AI Studio API.

## Features

- PDF upload and text extraction with `PyPDFLoader`
- Recursive chunking with overlap for retrieval quality
- Chroma vector store backed by Google embeddings
- Answer generation with Gemini
- Chat-style Q&A with retrieved passage previews
- Session state so the PDF index is not rebuilt on every question
- Responsive frosted-glass UI with document metrics and assistant panel

## Requirements

- Python 3.10 or newer
- A Google AI Studio API key

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If your project is inside OneDrive and package installation hits file locks,
create the virtual environment outside the synced folder and run Streamlit from
this project directory.

Set your key in the environment before starting Streamlit:

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

## Run

```powershell
streamlit run app.py
```

Then open the local URL Streamlit prints, usually:

```text
http://localhost:8501
```

## Workflow

1. Upload a PDF.
2. Click `Build index`.
3. Ask questions in the document chat.
4. Expand `Retrieved passages` under an answer to inspect the source context.
