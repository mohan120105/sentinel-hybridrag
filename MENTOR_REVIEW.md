# Sentinel GraphRAG - Mentor Review Notes

## 1. Project Objective
Sentinel is a banking governance GraphRAG prototype with two agentic pipelines:
- Curator Agent (ingestion): parse policy documents, map to ontology, and update supersession chains.
- Co-Pilot Agent (retrieval): answer user questions only from active policy context with evidence citation.

This PoC demonstrates strict governance behavior:
- superseded policies are excluded from answers,
- user responses are grounded in graph evidence,
- fallback response is returned when active evidence is missing.

---

## 2. Tech Stack Implemented
- Python 3.10+
- Neo4j (graph + vector index)
- LangChain + ChatGroq (`llama-3.3-70b-versatile`)
- HuggingFace sentence embeddings (`all-MiniLM-L6-v2`)
- Pydantic schemas for structured extraction
- Streamlit UI for module navigation and Co-Pilot chat

---

## 3. What We Built

### 3.1 `init_graph.py` (Graph Initialization + Ingestion)
Implemented:
- Ontology bootstrap (`Category` nodes):
  - `Retail_Loans`, `Corporate_Banking`, `KYC_AML`, `Credit_Cards`, `Tax_Compliance`
- Synthetic policy corpus (banking realistic examples):
  - AML 2024 PAN threshold (`INR 100,000`)
  - AML 2026 urgent update (`INR 50,000`) replacing 2024 memo
  - Retail NRI home loan policy (`8.35%`, up to `INR 20,000,000`)
- Curator schema (`GraphAction`) with strict fields:
  - `target_node`, `action_type`, `extracted_rule`, `superseded_document`
- Structured extraction via `with_structured_output(GraphAction)`
- Policy ingestion into Neo4j:
  - `(:Policy)-[:BELONGS_TO]->(:Category)`
  - supersession link: `(:Policy)-[:SUPERSEDES]->(:Policy)`
  - superseded node marked inactive (`active=false`)
- Vector retrieval support:
  - generated embeddings per policy
  - stored in `Policy.embedding`
  - vector index `policy_embeddings` created in Neo4j
- Robust error handling:
  - Neo4j connection/service errors
  - Neo4j query errors
  - LLM schema validation failures

### 3.2 `query_copilot.py` (Retrieval Agent)
Implemented:
- Environment normalization and `.env` loading
- Neo4j connection hardening:
  - local `neo4j://` auto-converted to `bolt://` for single-node setup
  - connectivity verification before runtime loop
- Vector retrieval query:
  - `db.index.vector.queryNodes('policy_embeddings', ...)`
  - strict active-policy filter: `WHERE NOT ()-[:SUPERSEDES]->(p)`
- LLM answer synthesis from retrieved context only
- strict fallback:
  - `I cannot find a verified active policy for this in the current database.`
- terminal interactive Q&A mode

### 3.3 `app.py` (Streamlit Integration)
Implemented:
- Sidebar navigation modules:
  - Dashboard
  - Curator Agent
  - Universal Ingestion
  - `💬 Co-Pilot (Retrieval)`
- Co-Pilot chat using:
  - `st.chat_input`
  - `st.chat_message`
- Cached resources to reduce lag:
  - Neo4j driver
  - Groq LLM
  - embeddings model
- UI evidence section with source citation expander
- relevance filtering before citation display to reduce noisy sources
- clear chat button and graceful error handling

---

## 4. Key Problems Encountered and How We Solved Them

### Issue A: "Unable to retrieve routing information"
Cause:
- Local Neo4j endpoint used routing URI (`neo4j://`) in a non-cluster setup.

Fix:
- Added local URI conversion to `bolt://` in retrieval driver builder.
- Added connectivity verification and better startup messages.

### Issue B: Connection refused (`WinError 10061`)
Cause:
- Neo4j service not listening on `127.0.0.1:7687` at that moment.

Fix:
- Added clear diagnostics and user-friendly startup failure guidance.
- Verified port/service externally before continuing.

### Issue C: Neo4j warning for unknown property `p.text`
Cause:
- Query referenced non-existent property name.

Fix:
- Standardized on `p.source_text` and removed `p.text` fallback in strict query path.

### Issue D: Source citation showing unrelated documents
Cause:
- Initial citation used full retrieved set or permissive lexical checks.

Fix:
- Added question-term filtering and overlap scoring in `app.py`.
- Removed generic stopwords and kept best-matching context for citation.

---

## 5. Current End-to-End Flow
1. Run ingestion (`init_graph.py`) to reset ontology and ingest policy docs.
2. Curator extraction maps each doc to ontology + supersession action.
3. Graph stores policy nodes, category edges, supersession lineage, embeddings.
4. Co-Pilot receives question and generates embedding.
5. Neo4j vector index retrieves semantic candidates.
6. Strict filter removes superseded policies.
7. LLM answers only from filtered active context.
8. Streamlit displays answer + source citation.

---

## 6. Demo Script for Mentor Review
Use these questions in Streamlit Co-Pilot:
1. `What is the PAN card limit for cash deposits?`
Expected:
- Answer from 2026 AML urgent circular
- PAN threshold should be INR 50,000
- Citation should point to AML circular category `KYC_AML`

2. `What is the interest rate for NRI home loans?`
Expected:
- Answer from Retail NRI home loan policy
- Interest 8.35% p.a.
- Citation should point to Retail Loans policy

3. `What is the card fee waiver in this database?`
Expected:
- strict fallback response if no active verified policy exists

---

## 7. How to Run
```powershell
# 1) install deps in venv
pip install -r requirements.txt

# 2) ingest graph data
python init_graph.py

# 3) run Streamlit app
streamlit run app.py
```

---

## 8. Current Limitations
- Dataset is still synthetic and small (3 docs).
- Citation filtering in app layer is lexical + overlap based; stronger reranking can be added.
- Multi-policy conflict resolution can be improved with explicit confidence thresholds.

---

## 9. Recommended Next Steps
1. Add policy version metadata (`effective_from`, `effective_to`, `regulator_ref`) to strengthen auditability.
2. Add explicit retrieval confidence threshold; return fallback if below threshold.
3. Add unit/integration tests for:
   - supersession chain behavior,
   - active-only retrieval constraint,
   - citation precision.
4. Add ingestion pipeline for real PDFs with chunk-level provenance references.
5. Introduce role-based access and policy domain restrictions for enterprise deployment.

---

## 10. Deliverables Created
- `init_graph.py`
- `query_copilot.py`
- `app.py`
- `requirements.txt`
- `MENTOR_REVIEW.md` (this document)

---

## 11. Post-Review Updates Added After This Document Was First Written

The project has advanced beyond the original Streamlit-only prototype. The following work was added after the initial mentor review note was created.

### 11.1 FastAPI Service Layer Added (`api.py`)
Implemented:
- standalone FastAPI backend for Sentinel Co-Pilot
- CORS-enabled API for frontend integration
- `POST /chat` endpoint for policy-grounded question answering
- `GET /sessions` endpoint to list stored chat sessions
- lazy initialization of:
  - Neo4j driver
  - Groq LLM client
  - embedding model

### 11.2 Persistent Session Memory in Neo4j
Implemented in `api.py`:
- persistent conversation memory using:
  - `(:Session {id})`
  - `(:Message {role, content, timestamp})`
- per-request retrieval of the last 4 conversation turns
- ordered session replay using timestamps
- write-back of user + assistant turns after every successful answer
- graceful degradation when answer generation succeeds but message persistence fails

### 11.3 True Hybrid GraphRAG Retrieval Upgrade (`query_copilot.py`)
Retrieval was upgraded from vector-only retrieval to hybrid retrieval.

Implemented:
- hybrid candidate generation using:
  - Neo4j vector index: `policy_embeddings`
  - Neo4j full-text index: `policy_keywords`
- score fusion of semantic similarity and BM25 keyword results
- governance firewall retained:
  - `WHERE NOT ()-[:SUPERSEDES]->(p)`
- multi-hop retrieval expansion to include:
  - `(:CustomerType)` via `APPLIES_TO`
  - `(:DocumentRequirement)` via `REQUIRES`
- evidence-rich active context returned as structured `ActivePolicy`
- fallback to vector-only retrieval when full-text index is unavailable

### 11.4 History-Aware Answer Generation
Implemented in `api.py`:
- conversation history injected into the final answer prompt
- active policy context still remains the only authoritative evidence base
- strict no-answer fallback preserved:
  - `I cannot find a verified active policy for this in the current database.`
- rate-limit aware fallback messaging for Groq API failures

### 11.5 React + Vite Frontend Added (`frontend/`)
Implemented:
- separate React frontend for Sentinel Co-Pilot
- Vite-based development setup
- modern chat interface with:
  - session sidebar
  - new chat creation
  - session switching
  - live typing indicator
  - markdown rendering for answers
  - GFM table rendering for structured policy outputs
  - collapsible citation panel with document/category/score
- backend integration against FastAPI on `http://localhost:8000`

UI stack added:
- React 18
- Vite
- Tailwind CSS
- `react-markdown`
- `remark-gfm`
- `lucide-react`

### 11.6 Multimodal Universal Ingestion Strengthened (`app.py`)
The Streamlit ingestion module was enhanced significantly.

Implemented:
- Gemini multimodal extraction for uploaded PDFs and images
- stricter extraction prompt requiring table/slab data to be preserved inside `extracted_rule`
- normalization layer to map common LLM key aliases into strict `GraphAction` schema fields
- direct ingestion from extracted multimodal payload into Neo4j graph + vector storage
- schema validation and improved error handling for JSON parsing, Gemini output, and Neo4j write failures

### 11.7 Baseline Bulk Seeding Pipeline Added (`seed_database.py`)
Implemented:
- automated scan of `v1_baseline_docs/v1-base`
- batch extraction of policy files (`.pdf`, `.png`) using Gemini multimodal input
- ontology-constrained baseline ingestion
- enforced `CREATE_NEW` baseline seeding behavior
- configurable Gemini model selection from environment
- basic rate-limit spacing between document ingestion calls

### 11.8 Dependency Footprint Expanded (`requirements.txt`)
Added backend/runtime dependencies for the newer stack:
- `fastapi`
- `uvicorn[standard]`
- `google-genai`

Existing graph / LLM / retrieval stack remains in use:
- Neo4j
- LangChain Core
- LangChain Groq
- LangChain HuggingFace
- sentence-transformers
- Pydantic
- Streamlit

### 11.9 Current Architecture After Updates
Sentinel now supports two UI/application paths:

1. Streamlit path
   - dashboard / ingestion / retrieval prototype in one app

2. FastAPI + React path
   - FastAPI backend for GraphRAG orchestration
   - React frontend for multi-session Co-Pilot experience

Current end-to-end flow for the newer stack:
1. User opens React Co-Pilot UI.
2. Frontend calls FastAPI `/chat` with `session_id` and `user_question`.
3. FastAPI loads recent session history from Neo4j.
4. Query is embedded and sent through hybrid retrieval.
5. Active-only governance filtering removes superseded policies.
6. Multi-hop context includes customer type and required-document relations.
7. Groq generates grounded answer from active context plus short conversation history.
8. User and assistant messages are persisted back into Neo4j.
9. Citations are returned to the frontend and rendered in the chat UI.

### 11.10 Updated Run Modes
Original Streamlit demo still works, but there is now an additional API + frontend mode.

#### Streamlit mode
```powershell
pip install -r requirements.txt
streamlit run app.py
```

#### FastAPI + React mode
```powershell
# backend
pip install -r requirements.txt
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# frontend
cd frontend
npm install
npm run dev
```

### 11.11 Updated Deliverables Since Initial Review
Additional deliverables now present:
- `api.py`
- `seed_database.py`
- `frontend/`
  - `src/App.jsx`
  - Vite + Tailwind setup files
- baseline policy corpus in `v1_baseline_docs/`

### 11.12 Summary of What Improved Since Initial Review
- Sentinel evolved from a Streamlit PoC into a two-tier application with API + frontend separation.
- Retrieval matured from strict vector-only search into hybrid GraphRAG with keyword fusion.
- Chat now supports persistent Neo4j-backed session memory.
- Ingestion now supports multimodal Gemini extraction and batch baseline seeding.
- Frontend UX is significantly stronger for live demo and product-style presentation.
