"""
Multilingual RAG sandbox

- Index English docs (plain text or PDF) from ../v1_baseline_docs
- Use intfloat/multilingual-e5-small for superior multilingual retrieval
- Store embeddings in a local ChromaDB (persisted under multilang_chroma_db)
- Accept non-English queries, retrieve English chunks, and ask an LLM to produce an answer in the user's language

Run:
    python multilang_sandbox/sandbox.py

Environment:
    - `GROQ_API_KEY` required: used to initialize ChatGroq for final answer generation

"""
import os
import glob
from typing import List
from dotenv import load_dotenv

# Use a lightweight local splitter to avoid langchain versioning issues
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    from groq import Groq
    HAS_GROQ = True
except Exception:
    HAS_GROQ = False

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_environment() -> None:
    """Load Groq credentials from the repository's .env first, then fallback locations."""

    load_dotenv(os.path.join(REPO_ROOT, ".env"), override=True)
    load_dotenv(os.path.join(os.path.dirname(__file__), ".env"), override=True)
    load_dotenv(override=False)

    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key is not None:
        os.environ["GROQ_API_KEY"] = groq_key.strip().strip('"').strip("'")


# Load environment variables from .env file(s)
load_environment()

SAMPLE_TEXT = (
    """
    Banking Policy: Customer Due Diligence
    All customers must provide valid identification. Banks should perform risk-based due diligence on transactions, and report suspicious activities per regulatory requirements. Sensitive customer data must be stored encrypted and access logged. Failure to comply may result in fines and operational restrictions.
    """
)

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "multilang_chroma_db")
# Use symmetric multilingual MiniLM for embeddings (dimension 384)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SIMILARITY_THRESHOLD = 0.70  # Minimum similarity score (0-1) to include retrieved documents


def load_documents_from_folder(folder: str) -> List[str]:
    """Load text from .txt files or extract text from PDFs in a folder.

    Falls back to SAMPLE_TEXT if nothing found.
    """
    texts = []
    txt_paths = glob.glob(os.path.join(folder, "**", "*.txt"), recursive=True)
    pdf_paths = glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)

    for p in txt_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except Exception:
            continue

    if PdfReader and pdf_paths:
        for p in pdf_paths:
            try:
                reader = PdfReader(p)
                pages = [page.extract_text() or "" for page in reader.pages]
                texts.append("\n".join(pages))
            except Exception:
                continue

    if not texts:
        texts = [SAMPLE_TEXT]

    return texts


def chunk_texts(texts: List[str], chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Simple splitter: splits on whitespace into overlapping chunks.

    This avoids depending on langchain's text_splitter submodule which
    may vary between releases.
    """
    docs: List[str] = []
    for t in texts:
        words = t.split()
        if not words:
            continue
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            docs.append(chunk)
            if end >= len(words):
                break
            start = end - chunk_overlap
    return docs


class SimpleDoc:
    def __init__(self, text: str, metadata: dict):
        self.page_content = text
        self.metadata = metadata


class LocalVectorStore:
    def __init__(self, collection, model: SentenceTransformer):
        self.collection = collection
        self.model = model

    def similarity_search(self, query: str, k: int = 3, similarity_threshold: float = None):
        if similarity_threshold is None:
            similarity_threshold = SIMILARITY_THRESHOLD
        # Symmetric MiniLM: embed raw query without E5 prefix
        q_emb = self.model.encode([query], convert_to_numpy=True)[0].tolist()
        res = self.collection.query(query_embeddings=[q_emb], n_results=k, include=["documents", "metadatas", "distances"])
        docs = []
        
        # Get results for the first (and only) query
        docs_list = res.get("documents", [[]])[0] if res.get("documents") else []
        metas_list = res.get("metadatas", [[]])[0] if res.get("metadatas") else []
        dists_list = res.get("distances", [[]])[0] if res.get("distances") else []
        
        for d, m, dist in zip(docs_list, metas_list, dists_list):
            # Convert distance to similarity (1 - cosine_distance)
            similarity = 1 - dist
            if similarity >= similarity_threshold:
                docs.append(SimpleDoc(d, m))
        return docs


def build_or_load_vectorstore(texts: List[str], persist_dir: str = PERSIST_DIR) -> LocalVectorStore:
    """Create or load a chroma vectorstore persisted on disk using sentence-transformers.
    Returns a LocalVectorStore wrapper with `similarity_search(query, k)` method.
    """
    os.makedirs(persist_dir, exist_ok=True)
    model = SentenceTransformer(EMBEDDING_MODEL)

    client = chromadb.Client()

    # get or create collection
    try:
        collection = client.get_collection(name="multilang")
    except Exception:
        collection = client.create_collection(name="multilang")

    # if collection empty, populate
    try:
        count = collection.count()
    except Exception:
        # older/newer APIs may not have count(); try to check documents
        count = 0

    if count == 0:
        docs = chunk_texts(texts)
        ids = [f"doc_{i}" for i in range(len(docs))]
        metadatas = [{"source": f"doc_{i}"} for i in range(len(docs))]
        # Symmetric MiniLM: embed document chunks directly
        embeddings = model.encode(docs, show_progress_bar=False, convert_to_numpy=True)
        # chromadb expects lists
        emb_lists = [e.tolist() for e in embeddings]
        collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=emb_lists)
        try:
            client.persist()
        except Exception:
            pass

    return LocalVectorStore(collection, model)


def multilingual_query(vectordb, query: str, k: int = 3, user_lang: str = None) -> str:
    """Accepts a non-English `query`, returns an answer in the same language.

    Workflow:
    1. Similarity search in English index using multilingual embeddings
    2. Retrieve top-k English chunks
    3. Ask an LLM (if available) to answer in user's language using the English context
    """
    results = vectordb.similarity_search(query, k=k)
    contexts = "\n---\n".join([r.page_content for r in results])

    print("\n[RETRIEVED CONTEXT]")
    print(contexts)
    print("[END CONTEXT]\n")

    system_prompt = (
        "You are a compliance assistant. Use the provided English context to answer the user's question. "
        "Respond in the user's original language exactly, keep answers concise and suitable for regulatory review."
    )

    human_prompt = f"User question (in their language): {query}\n\nContext (English passages):\n{contexts}\n\nAnswer in the user's language."

    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key and not groq_api_key.startswith("gsk_"):
        return (
            "[Groq API Error] GROQ_API_KEY is present but does not look valid. "
            "Put the real key in c:/Users/MOHAN/Documents/Bank-rag/.env as GROQ_API_KEY=gsk_..."
            f"\n\nContext available:\n{contexts}"
        )

    if HAS_GROQ and groq_api_key:
        try:
            client = Groq(api_key=groq_api_key)
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": human_prompt}
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_text = str(e).lower()
            if "invalid_api_key" in error_text or "401" in error_text:
                return (
                    "[Groq API Error] GROQ_API_KEY from your .env is invalid. "
                    "Replace it with a valid Groq key (usually starts with gsk_) and rerun the sandbox."
                    f"\n\nContext available:\n{contexts}"
                )
            return f"[Groq API Error] {str(e)}\n\nContext available:\n{contexts}"

    # Placeholder fallback: return the English context and indicate where to plug an LLM
    placeholder = (
        "[LLM not configured] Retrieved English context passages: \n"
        f"{contexts}\n\nResponding in user's language requires an LLM (set GROQ_API_KEY)."
    )
    return placeholder


def main():
    docs_folder = os.path.join(REPO_ROOT, "v1_baseline_docs")

    texts = load_documents_from_folder(docs_folder)
    vectordb = build_or_load_vectorstore(texts)

    examples = [
        ("मुझे ग्राहक पहचान आवश्यकताओं के बारे में बताएं।", "hi"),  # Hindi
        ("¿Qué requisitos existen para reportar actividades sospechosas?", "es"),  # Spanish
        ("గ诺లు డౌట్: వినియోగదారు నిర్ధారణ విధానం ఏమిటి?", "te"),  # Telugu-ish example
    ]

    for q, lang in examples:
        print("\n---\nQuery:", q)
        ans = multilingual_query(vectordb, q, k=3, user_lang=lang)
        print("Answer:\n", ans)


if __name__ == "__main__":
    main()
