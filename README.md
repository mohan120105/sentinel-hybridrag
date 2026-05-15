# Sentinel GraphRAG: Enterprise Conversational AI for Banking Compliance

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20Database-blue)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![Groq](https://img.shields.io/badge/Groq-Llama--3.3-black)
![Gemini](https://img.shields.io/badge/Gemini-Vision%20Multimodal-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Async%20API-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B)

**Sentinel GraphRAG** is a production-grade conversational AI system for banking policy compliance. It combines multimodal document ingestion, hybrid vector + keyword retrieval, Neo4j graph governance, multilingual support, and deterministic audit trails to deliver hallucination-resistant, compliance-aware Q&A with full regulatory traceability.

## The Challenge: Enterprise RAG for Regulated Domains

1. **Policy Versioning**: Vector stores retrieve both active and superseded policies → conflicting answers
2. **Numeric Fidelity**: OCR pipelines flatten complex tables/slabs → loss of critical thresholds (e.g., "TDS rates", "loan caps")
3. **Semantic Ambiguity**: "What documents are needed?" has no single truth without multi-hop context (customer type, policy version, etc.)
4. **Compliance Auditability**: Analysts must trace every answer to source policy + retrieval confidence → no confidence percentages in standard RAG
5. **Multilingual Enterprise**: Banking customers span Hindi, Telugu, Tamil, etc. → responses must be localized

## Sentinel Solution: Governance-First Hybrid Retrieval

Sentinel implements a **three-layer retrieval engine** with governance enforcement:

1. **Vector Search** (HuggingFace Embedding Space): Captures semantic intent via `paraphrase-multilingual-MiniLM-L12-v2` (384-dim vectors)
2. **Full-Text Search** (Neo4j BM25 Index): Boosts exact keyword + identifier matches (policy names, acronyms like "KYC", "TDS")
3. **Hybrid Score Fusion**: `combined_score = vector_score + (BM25_score / 10.0)` for balanced relevance
4. **GLAC Access Control**: `WHERE ($user_tier = 1 OR p.access_code = 2)` enforces tier-based policy access
5. **Version Governance**: `WHERE NOT ()-[:SUPERSEDES]->(p)` filters active policies only
6. **Multi-Hop Resolution**: :BELONGS_TO, :APPLIES_TO, :REQUIRES edges provide customer segment + document context

## Core Features

### 🔄 Ingestion Pipeline

- **Multimodal Extraction**: Gemini Vision API parses PDFs/images with **table preservation** (critical for numeric policies)
- **Pydantic Validation**: Structured `GraphAction` model enforces schema correctness (target_node, action_type, extracted_rule, applies_to_customer, requires_document)
- **Ontology Guardrails**: Fixed 10-category taxonomy (Retail_Loans, Corporate_Banking, KYC_AML, Tax_Compliance, etc.) prevents hallucinated classifications
- **Policy Lineage**: :SUPERSEDES edges track version history without hard deletes

### 🔍 Hybrid Retrieval Engine

- **True Hybrid Search**: Vector + BM25 score fusion for semantic + keyword relevance
- **GLAC Governance**: Employee tier-based access control (Admin: all policies, Operator: public tier-2 only)
- **Active Policy Filtering**: Automatically excludes superseded policies from retrieval
- **Confidence Scoring**: Raw similarity scores normalized to UI-ready percentages (0-98.5%)
- **Fallback Robustness**: Vector-only retrieval if full-text index unavailable
- **Interactive Citation Graph**: Click-to-explore evidence relationships (policy → category → customer type → required docs) via force-directed visualization

### 🌍 Multilingual Support

- **Language Detection**: FastText (primary, 50% confidence threshold) + langdetect fallback
- **Supported Languages**: English, Hindi, Telugu, Tamil, Spanish, French, German, Portuguese, Arabic, Bengali, Punjabi, Marathi, Kannada, Malayalam, Gujarati, Urdu, Chinese
- **Response Localization**: LLM response enforced in detected user language
- **Technical Term Preservation**: Acronyms (TDS, KYC, AML) remain in English for regulatory clarity

### 💼 Executive Auditor Persona (High-Value Exposure Detection) – Stage 3

- **Numeric Grounding**: System prompt instructs the LLM to extract and preserve exact numeric values (TDS rates, liquidity ratios, financial thresholds)
- **High-Value Risk Flagging**: Regex-based `detect_high_value_exposure()` scans context for INR amounts ≥ ₹10 Crore and flags as **[CRITICAL TIER 1 RISK]**
- **Automatic Bolding**: Amounts exceeding ₹10 Crore are automatically bold-formatted in responses with source document attribution
- **Business Value**: Executives and compliance officers immediately identify enterprise-level financial exposures without manual review
- **Example**: Query about corporate lending limits → Response flags **₹14.5 Crore cap** with policy name → Operator can escalate to risk committee

### 💬 Conversation & Session Memory

- **Neo4j Session Persistence**: Last 4 chat messages injected for conversation continuity
- **Audit Trail**: Every turn saved with user_tier, retrieval_tier, citations, sentinel_reasoning
- **Citations with Confidence**: Evidence snapshot includes document name, category, customer types, required docs
- **Deterministic Replay**: Session history replayable for compliance audits

### 🚀 Enterprise Features

- **Microservice Architecture**: Embeddings offloaded to HF Spaces (no local GGUF models)
- **Graceful Degradation**: FastText unavailable → langdetect → English; embedding service down → 503 error (no silent fallback)
- **Time-Budgeted Features**: Follow-up suggestions respect 2.5-second timeout with ThreadPoolExecutor
- **Lazy Singletons**: LLM, embeddings, driver initialized on first request (no startup coupling)
- **Comprehensive Logging**: All errors, warnings, and fallback events logged to console
- **Secure Asset Proxy**: Private GitHub policy vault with GLAC-enforced read access
- **Policy Repository Tab**: Browsable UI for policy discovery and filtered access

## System Architecture

### Data Flow Layers

```
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION PIPELINE                                              │
│  PDF/Image → Gemini Vision → GraphAction Pydantic → Neo4j       │
│  - Multimodal extraction with table preservation                 │
│  - Ontology-constrained classification                           │
│  - Policy lineage via :SUPERSEDES edges                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  GRAPH SYNTHESIS                                                  │
│  Create: Policy, Category, CustomerType, DocumentRequirement     │
│  Edges: :BELONGS_TO, :APPLIES_TO, :REQUIRES, :SUPERSEDES       │
│  Indexes: Vector (cosine), Full-Text (BM25)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  QUERY PROCESSING                                                 │
│  1. Language Detection (FastText → langdetect)                   │
│  2. Question Embedding (HF Spaces: paraphrase-multilingual)      │
│  3. Hybrid Retrieval (vector + text fusion + GLAC + versioning)  │
│  4. Retrieval Tier Classification (exact/partial/no-match)       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  RESPONSE SYNTHESIS                                               │
│  History-aware LLM (Groq Llama-3.3-70b) + Tiered Response        │
│  - Exact match: Direct answer                                    │
│  - Partial match: Acknowledge gap, provide related info          │
│  - No match: "I cannot find verified policy..."                  │
│  - Multilingual output in detected user language                 │
│  - Citations with confidence scores                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  AUDIT TRAIL PERSISTENCE                                          │
│  Neo4j Session: user message → assistant message                 │
│  Metadata: tier, retrieval_tier, citations_json, reasoning       │
│  Replayable: Full session history for compliance review          │
└─────────────────────────────────────────────────────────────────┘
```

### Vector Search: Asymmetric Prefixing

```python
# Document (ingestion time)
embedding = embed(f"passage: {rule}\n\n{source_text}")

# Query (retrieval time)
question_embedding = embed(f"query: {user_question}")

# Result: Asymmetric space optimized for search intent
```

### Multi-Hop Evidence Graph

Sentinel's graph structure connects each policy to its context:

```
Policy (name, embedding, extracted_rule, active, access_code, issue_date)
    ├─ :BELONGS_TO → Category (Retail_Loans, Corporate_Banking, etc.)
    ├─ :APPLIES_TO → CustomerType (NRI, MSME, Large_Corp, etc.)
    ├─ :REQUIRES → DocumentRequirement (Passport, Board_Resolution, etc.)
    └─ :SUPERSEDES → [old_policy] (version lineage, marks old as inactive)
```

## Repository Structure

```
sentinel-graphrag/
├── api.py                          # FastAPI control plane (chat, ingest, sessions, follow-ups)
├── app.py                          # Streamlit UI for dashboard + ingestion
├── connect.py                      # Neo4j driver factory
├── init_graph.py                   # Ontology bootstrap + index creation
├── query_copilot.py                # Core retrieval engine (hybrid search, language detection)
├── seed_database.py                # Baseline document ingestion (v1_baseline_docs/ PDFs)
├── prompt_modifier.py              # Optional HF Spaces edge prompt enhancement
├── test_detection.py               # Language detection test suite
├── test_detection_minimal.py       # Minimal language detection demo
├── requirements.txt                # Python dependencies
├── SYSTEM_AUDIT.md                 # Comprehensive technical audit (all functions)
├── README.md                       # This file
├── docs/
│   └── FASTTEXT_MODEL_INSTRUCTIONS.md    # FastText language model setup
├── frontend/                       # Vite React UI (optional modern frontend)
│   ├── src/
│   │   ├── App.jsx                 # Main React component
│   │   ├── CitationMap.jsx         # Evidence graph visualization
│   │   ├── SentinelReasoning.jsx   # Retrieval tier & reasoning display
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── models/
│   ├── google_gemma-3-1b-it-Q4_K_M.gguf    # (optional local LLM)
│   └── lid.176.ftz                         # FastText language ID model
├── v1_baseline_docs/               # Seed policy documents (PDFs/PNGs)
└── img_data/                       # UI screenshot assets
```

### Key Files Explained

| File                       | Purpose               | Key Functions                                                                         |
| -------------------------- | --------------------- | ------------------------------------------------------------------------------------- |
| **api.py**           | FastAPI control plane | `chat()`, `ingest_document()`, `/sessions`, `/enhance`                        |
| **query_copilot.py** | Retrieval core        | `retrieve_active_policy()`, `detect_user_language()`, `generate_answer()`       |
| **init_graph.py**    | Neo4j setup           | `initialize_ontology()`, `create_policy_vector_index()`, `process_and_ingest()` |
| **app.py**           | Streamlit UI          | Session management, document upload, policy exploration                               |
| **seed_database.py** | Baseline ingestion    | Automated baseline policy seeding via Gemini + Pydantic                               |
| **connect.py**       | Neo4j factory         | `build_neo4j_driver()` (connection management)                                      |

## Quickstart

### Prerequisites

- **Python 3.10+**
- **Neo4j** (local or cloud instance)
- **API Keys** (free tier available):
  - Groq (for Llama-3.3-70b inference): [console.groq.com](https://console.groq.com)
  - Google Gemini (for multimodal extraction): [ai.google.dev](https://ai.google.dev)
  - HuggingFace (for embedding space + optional router): [huggingface.co](https://huggingface.co)
- **Optional**: FastText language model (`lid.176.ftz`) — auto-downloaded if missing

### 1. Clone & Setup Environment

```bash
git clone https://github.com/mohan1201/sentinel-graphrag.git
cd sentinel-graphrag
python -m venv venv
```

**Activate virtual environment:**

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
source venv/bin/activate
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with all required keys:

```dotenv
# Neo4j Graph Database
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# LLM Inference (Groq)
GROQ_API_KEY=your_groq_api_key

# Multimodal Extraction (Google Gemini)
GEMINI_API_KEY=your_gemini_api_key

# Embeddings (HuggingFace Spaces)
HF_EMBEDDING_SPACE=mohan1201/sentinel-embedding-server
HF_EMBEDDING_API_NAME=/embed
HF_TOKEN=your_huggingface_token

# Optional: Prompt Enhancement via HF Router
HF_ROUTER_URL=https://your-hf-router-space.hf.space/run/predict
HF_API_TOKEN=your_huggingface_api_token

# Optional: Similarity Filtering via HF
HF_SIMILARITY_ENDPOINT=https://your-similarity-space.hf.space/run/predict

# GitHub Document Vault (Stage 3: Secure Asset Proxy)
GITHUB_REPO=mohan120105/JatayuS5-TuringMachines
GITHUB_POLICY_MANIFEST_REPO=mohan120105/JatayuS5-TuringMachines
GITHUB_POLICY_CONTENTS_REPO=mohan120105/JatayuS5-TuringMachines
GITHUB_TOKEN=your_github_pat_token
GITHUB_DOCS_ROOT=hackathon-docs
GITHUB_POLICY_MANIFEST_PATH=policy_access_manifest.json

# Feature Flags
ENABLE_FOLLOWUP_SUGGESTIONS=true

# Language Detection Model (auto-discovered if missing)
FASTTEXT_LANG_MODEL=./models/lid.176.ftz
```

### 3. Bootstrap Neo4j Database

**Option A: Manual Setup (Recommended)**

```bash
python init_graph.py
```

This creates:

- 10 Policy Category nodes (fixed ontology)
- Vector index (cosine similarity, 384-dim)
- Full-text index (BM25 on name, rule, source_text)
- Synthetic policy documents for testing

**Option B: Seed Baseline Documents**

```bash
python seed_database.py
```

Automatically ingests all PDFs/PNGs from `v1_baseline_docs/` via Gemini extraction.

### 4. Run the Application

**Option A: FastAPI (Production)**

```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

API available at: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

**Option B: Streamlit UI (Development)**

```bash
streamlit run app.py
```

UI available at: `http://localhost:8501`

**Option C: CLI Interactive Mode**

```bash
python query_copilot.py
```

Query the graph from terminal interactively.

### 5. Test Language Detection

```bash
python test_detection.py
```

Tests multilingual support (English, Hindi, Telugu, Spanish, etc.).

## API Endpoints

### Chat Endpoint

```http
POST /chat
Content-Type: application/json

{
  "session_id": "uuid-string",
  "user_question": "What are the NRI home loan limits?",
  "employee_id": "1234"  # Prefix 1=Admin, 2=Operator, 3=Viewer
}

Response:
{
  "answer": "NRI applicants can borrow up to INR 20,000,000 subject to credit approval.",
  "citations": [
    {
      "document_name": "Retail_NRI_Home_Loan_Policy",
      "category": "Retail_Loans",
      "raw_score": 0.92,
      "match_confidence": 95.2
    }
  ],
  "retrieval_tier": "exact_match",
  "sentinel_reasoning": "Direct evidence found in Retail_NRI_Home_Loan_Policy...",
  "followup_suggestions": [...]
}
```

### Ingest Document Endpoint

```http
POST /ingest
Content-Type: multipart/form-data

file: <PDF or image>
employee_id: 1234
access_code: 2  # 1=Admin-only, 2=Public (visible to all tiers)

Response:
{
  "message": "Policy ingested successfully",
  "document_name": "Tax_Compliance_2026_Update",
  "extracted_rule": "TDS rate on salary: 10% for income ≤ 5L, 20% for > 5L"
}
```

### Session Management

```http
GET /sessions?employee_id=1234
GET /sessions/{session_id}/messages?employee_id=1234
```

### Prompt Enhancement (Optional)

```http
POST /enhance
Content-Type: application/json

{
  "user_input": "nri docs"
}

Response:
{
  "enhanced_prompt": "What documents are required for NRI customers?"
}
```

### Policy Repository (GitHub Vault Proxy) – Stage 3

Retrieves policies from a private GitHub repository with tier-based GLAC access control.

**List Available Policies:**

```http
GET /api/v1/policies?employee_id=1234
Authorization: Bearer {token_if_needed}

Response:
[
  {
    "name": "Retail_Loans_2026",
    "category": "Retail_Loans",
    "access_code": 2,
    "url": "/api/v1/policies/view/Retail_Loans_2026.md"
  },
  {
    "name": "Corporate_Banking_Policy",
    "category": "Corporate_Banking",
    "access_code": 1,
    "url": "/api/v1/policies/view/Corporate_Banking_Policy.md"
  }
]
```

**Note:** Only policies accessible by the user's tier are returned.
- Tier 1 (Admin, prefix `1***`): sees all policies (access_code = 1 or 2)
- Tier 2–3 (Operator/Viewer, prefix `2***` or `3***`): sees public policies only (access_code = 2)

**View Policy File:**

```http
GET /api/v1/policies/view/Retail_Loans_2026.md?employee_id=1234

Response: (File content with GLAC filtering applied)
200 OK – File content returned
403 Forbidden – User tier insufficient for requested policy
404 Not Found – Policy not in vault
```

**Why This Matters (Secure Asset Proxy Pattern):**
- **Private Policy Storage**: Policies live in a private GitHub repository (immutable, audit-ready, version-controlled)
- **Zero-Trust Retrieval**: Every `/api/v1/policies/view` call validates user tier + access_code before returning content
- **Deterministic Lineage**: Policy version control via Git history; no local corruption risk
- **End-to-End Governance**: Integrates seamlessly with Sentinel's retrieval engine for hybrid graph + vault queries

**Environment Setup:**

```bash
# Generate GitHub PAT (Personal Access Token) with read:repo permission
# https://github.com/settings/tokens

# Set in .env:
GITHUB_REPO=mohan120105/JatayuS5-TuringMachines
GITHUB_TOKEN=ghp_xxxxxxxxxxxx

# Create policy_access_manifest.json in your repo root:
{
  "policies": [
    {"name": "Retail_Loans_2026.md", "access_code": 2},
    {"name": "Corporate_Banking_Policy.md", "access_code": 1}
  ]
}
```

## Governance Patterns

### Active Policy Filtering

Returns only policies not superseded by newer versions:

```cypher
MATCH (p:Policy)
WHERE NOT ()-[:SUPERSEDES]->(p)
RETURN p
```

### GLAC Access Control

Enforces employee tier-based policy visibility:

```cypher
WHERE ($user_tier = 1 OR p.access_code = 2)
-- Tier 1 (Admin, prefix 1***): sees all policies (access_code = 1 or 2)
-- Tier 2 (Operator, prefix 2***): sees public policies only (access_code = 2)
-- Tier 3 (Viewer, prefix 3***): sees public policies only (access_code = 2)
```

**Access Code Meanings:**
- `access_code = 1`: Admin-only (restricted to Tier 1 users only)
- `access_code = 2`: Public (visible to all employee tiers)

### Multi-Hop Relationship Resolution

Retrieves customer applicability and required documents:

```cypher
OPTIONAL MATCH (p)-[:APPLIES_TO]->(ct:CustomerType)
OPTIONAL MATCH (p)-[:REQUIRES]->(dr:DocumentRequirement)
RETURN collect(DISTINCT ct.name) AS customer_types,
       collect(DISTINCT dr.name) AS required_docs
```

## Usage Examples

### Example 1: Banking Policy Query (English)

```
User: "What is the TDS rate on salary deposits?"

System:
1. Detects: English (confidence: 98%)
2. Embeds: "query: What is the TDS rate on salary deposits?"
3. Retrieves: Tax_Compliance_2026_Update (score: 0.88)
4. Generates: "According to Tax_Compliance_2026_Update, the TDS rate 
              is 10% for income ≤ 5 Lakhs and 20% for income above 5 Lakhs."
5. Citations: [Tax_Compliance_2026_Update | Category: Tax_Compliance | Score: 88%]
```

### Example 2: Multilingual Query (Hindi)

```
User: "NRI ग्राहकों के लिए आवश्यक दस्तावेज़ कौन से हैं?"

System:
1. Detects: Hindi (confidence: 95%)
2. Embeds: "query: NRI ग्राहकों के लिए आवश्यक दस्तावेज़..."
3. Retrieves: Retail_NRI_Home_Loan_Policy (score: 0.92)
4. Generates: (Response in Hindi)
   "NRI ग्राहकों को पासपोर्ट कॉपी, विदेशी पता प्रमाण, और वेतन पर्ची प्रदान करनी चाहिए।"
5. Citations: [Retail_NRI_Home_Loan_Policy | Category: Retail_Loans | Score: 92%]
```

### Example 3: Partial Match (Retrieval Tier Classification)

```
User: "What is the audit frequency for corporate accounts?"

System:
1. Retrieves: Risk_Management_Policy (score: 0.55, partial semantic overlap)
2. Classifies: PARTIAL_MATCH
3. Generates: "I found documentation regarding corporate risk management, 
              but it does not specifically state the audit frequency. 
              However, based on available policy: [Summarizes related info]"
4. Suggests: ["What audit requirements apply to corporate accounts?", 
              "How often must risk reviews occur?", ...]
```

## Performance Characteristics

| Component                    | Latency         | Notes                                              |
| ---------------------------- | --------------- | -------------------------------------------------- |
| Language Detection           | 10-50ms         | FastText or langdetect lookup                      |
| Query Embedding              | 100-500ms       | HF Spaces network latency                          |
| Hybrid Retrieval             | 50-200ms        | Neo4j vector + text index + relationship traversal |
| LLM Generation               | 2-10s           | Groq Llama-3.3-70b inference                       |
| Session History Fetch        | 10-50ms         | Neo4j last 4 messages query                        |
| **Total E2E (cached)** | **3-12s** | Dominated by LLM inference                         |

**Performance SLA Notes:**

- **Query Processing (<1.5s):** Language detection + embedding + retrieval combined, excluding LLM time
- **Follow-up Suggestion Budget:** 2.5-second hard timeout via ThreadPoolExecutor (gracefully skips if overrun)
- **Embedding Service Timeout:** 60 seconds default; override via code if HF Spaces is congested
- **p95/p99 Latency:** Typical p95 = 8-10s (LLM-dominated); p99 = 12-15s under load
- **SLA Targets:**
  - `exact_match` retrieval tier: <5s e2e (high confidence retrieval)
  - `partial_match` retrieval tier: <12s e2e (lower confidence, suggestions included)
  - `no_match`: <2s (strict no-answer fallback)
- **Under Peak Load:** Query queuing may extend latency; consider horizontal scaling of Neo4j + API replicas

## Troubleshooting

### Neo4j Connection Fails

```bash
# Verify Neo4j is running and Bolt is enabled
neo4j status
# Default: bolt://127.0.0.1:7687
```

### FastText Model Not Found

```bash
# Automatically downloads on first language detection
# Or manually: docs/FASTTEXT_MODEL_INSTRUCTIONS.md
```

### Embedding Space Timeout

```bash
# HF_EMBEDDING_SPACE might be overloaded; increase timeout
# In query_copilot.py: timeout=60 (default)
# Or configure alternative embedding space
```

### Groq Rate Limit (429)

```bash
# Groq free tier: ~30 requests/minute
# Response includes: "Groq API rate limit encountered. Please retry in a few seconds."
```

## Testing

### Language Detection Tests

```bash
python test_detection.py        # Full test suite
python test_detection_minimal.py  # Quick demo
```

### Manual Integration Test

```bash
# Terminal 1: Start API
python -m uvicorn api:app --reload

# Terminal 2: Interactive CLI
python query_copilot.py

# Terminal 3: UI
streamlit run app.py
```

## Compliance & Auditability

Sentinel is designed for **regulated enterprises** (banking, finance, insurance):

- **Deterministic Responses**: Same question + same policies → same answer (replayable)
- **Full Audit Trail**: Session history persisted with citations, tier, reasoning
- **Evidence Traceability**: Every answer linked to source policy + confidence score
- **Version Governance**: Policy supersession tracked explicitly (no silent updates)
- **Access Control Audit**: Every retrieval tagged with user_tier for compliance replay
- **Compliance Logging**: All errors and fallbacks logged (no silent failures)

## Documentation

- **[SYSTEM_AUDIT.md](SYSTEM_AUDIT.md)** – Complete technical audit of all functions, data flows, and enterprise patterns
- **[docs/FASTTEXT_MODEL_INSTRUCTIONS.md](docs/FASTTEXT_MODEL_INSTRUCTIONS.md)** – FastText language model setup

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Make changes** and test: `python test_detection.py` (for language features)
4. **Document** your changes (add docstrings, update README)
5. **Commit** with clear messages: `git commit -m "Add feature X for Y"`
6. **Push** to your fork: `git push origin feature/your-feature`
7. **Open** a pull request with description of changes

## License

This project is provided as-is for research and enterprise deployment.

## Support & Questions

For issues, questions, or suggestions:

- Open an **Issue** on GitHub
- Check **SYSTEM_AUDIT.md** for technical deep-dives
- Review **test files** for usage examples

---

**Sentinel GraphRAG** — Enterprise Hybrid Retrieval for Banking Compliance
*Built for accuracy, auditability, and regulatory resilience.*
