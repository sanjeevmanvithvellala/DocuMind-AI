# DocuMind AI — PDF Summarization & Question Answering

## Features

- Upload and extract text from a PDF, preserving page numbers.
- Generate short, standard, or detailed summaries.
- Extract key points and terminology.
- Ask questions grounded in retrieved document passages.
- Generate MCQs and important exam-preparation questions.
- View extracted text by page.
- Download summaries, Q&A history, questions, and extracted text.
- Retrieval-augmented generation (RAG) using TF-IDF retrieval and Gemini.
- Prompt safeguards that tell the model to rely on document excerpts and admit when evidence is insufficient.

## Requirements

- Python 3.10 or newer recommended.
- Gemini API key for generative features.
- Internet access for the Gemini API.
- Text-based PDFs. Scanned PDFs need OCR before upload.

## Run locally

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` and replace `your_api_key_here` with your Gemini API key.

Then run:

```powershell
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`, then run:

```bash
streamlit run app.py
```

## API key setup

Create a Gemini API key using Google AI Studio. Keep it private. Never commit `.env` or paste the key into source code. The app also supports entering a key in the sidebar for the current session.

The model defaults to `gemini-3.8-flash`. If this model is unavailable for your account, set `GEMINI_MODEL` to a model currently supported by your Gemini API access.

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository, excluding `.env`.
2. Create a Streamlit Community Cloud app pointing to `app.py`.
3. Add `GEMINI_API_KEY` under the app's Secrets settings:
   ```toml
   GEMINI_API_KEY = "your-real-key"
   GEMINI_MODEL = "gemini-3.8-flash"
   ```
4. Deploy.

Do not commit real API keys.

## Architecture

1. **PDF extraction:** PyMuPDF extracts text page by page.
2. **Chunking:** page text is split into overlapping chunks while retaining page numbers.
3. **Retrieval:** TF-IDF ranks chunks relevant to the user's question.
4. **Generation:** Gemini produces answers from retrieved excerpts.
5. **Grounding:** prompts require citations to supplied page numbers and an explicit uncertainty response when evidence is insufficient.
6. **Study tools:** the same grounded workflow creates summaries, MCQs, and exam questions.

## Limitations

- Text extraction may fail on scanned/image-only PDFs; OCR is not built into this version.
- TF-IDF retrieval is lightweight and works best when query terms overlap with document wording. It is not a semantic embedding index.
- Summaries use selected representative passages for long documents; check the original PDF for complete coverage.
- Model-generated answers may still be incorrect. Page references are intended to help verification, not guarantee correctness.
- Gemini API usage may incur costs and is subject to account quotas and provider policies.
- Avoid uploading sensitive or confidential documents unless you have permission and have reviewed the provider's data policies.

## Assignment alignment

This app demonstrates:
- Transformers / LLMs
- GPT-style generative workflows (using Gemini as the configured LLM provider)
- Prompt engineering
- Text summarization and transformation
- Question answering
- API integration

## Suggested evaluation

Test with at least three PDFs of different lengths. For each, check:
- Summary coverage and factual accuracy.
- Whether answers match the cited pages.
- Whether the system correctly admits when an answer is absent.
- Relevance and correctness of generated MCQs.
- Extraction quality for multi-column and table-heavy PDFs.
