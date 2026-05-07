# Multilingual RAG Sandbox

This sandbox demonstrates cross-lingual semantic search using multilingual sentence embeddings and a local ChromaDB vector store. English documents are indexed and queries in other languages (Hindi, Telugu, Spanish, etc.) retrieve relevant English passages without explicit translation.

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r multilang_sandbox/requirements.txt
```

2. (Optional) If you have PDFs you want indexed, place them under `v1_baseline_docs/` (already present in the repo).

3. Set your Groq API key in your environment or `.env` file:

```powershell
$env:GROQ_API_KEY="your-groq-api-key-here"
```

4. Run the sandbox:

```powershell
python multilang_sandbox/sandbox.py
```

Notes
- The script uses `intfloat/multilingual-e5-small` for superior multilingual semantic retrieval.
- The script uses `ChatGroq` with the `llama-3.3-70b-versatile` model to generate multilingual answers.
- Ensure `GROQ_API_KEY` is exported or defined in a `.env` file in your project root.
- The vector store is automatically rebuilt when you switch embedding models.

Files
- `sandbox.py`: main script (indexing, query flow, example queries).

Why multilingual embeddings?

Multilingual embeddings map semantically similar sentences from different languages into a shared vector space. This avoids brittle translation steps and preserves nuance and domain-specific phrasing, which is critical for compliance and regulatory search systems. See `sandbox.py` comments for more discussion.
