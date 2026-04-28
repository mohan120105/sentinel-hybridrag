
# Sentinel: Enterprise Hybrid GraphRAG and Multi-Agent Governance

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20Database-blue)
![LangChain](https://img.shields.io/badge/LangChain-Framework-green)
![Groq](https://img.shields.io/badge/Groq-Llama--3-black)
![Gemini](https://img.shields.io/badge/Gemini-Vision%20Multimodal-orange)

Sentinel is an enterprise-grade Hybrid GraphRAG architecture designed to reduce LLM hallucinations in highly regulated domains such as banking and finance. It combines multimodal ingestion, vector retrieval, keyword search, and graph-based governance so responses are accurate, auditable, and compliance-aware.

## The Problem: Why Standard RAG Fails in Enterprise

1. Policy Churn (legacy data trap): Typical vector stores can retrieve both current and superseded policies, causing mixed or invalid answers.
2. Loss of Granularity: Traditional OCR pipelines often flatten complex banking tables, losing critical numeric context.
3. Flat Context: Real policy rules are relational and multi-hop. Flat retrieval cannot consistently resolve dependencies.

## Sentinel Solution

Sentinel uses a three-layer retrieval architecture:

- Vector Search (HuggingFace): captures semantic intent.
- Full-Text BM25 (Lucene): boosts exact keyword and identifier matches.
- Graph Traversal (Neo4j): executes governance logic to filter superseded content and resolve connected entities.

## Key Features

- Multimodal Pydantic Extraction: Uses Gemini Vision to parse complex tables into structured JSON while preserving markdown content.
- SUPERSEDES Firewall: Cypher guardrail (`WHERE NOT ()-[:SUPERSEDES]->(p)`) prevents outdated policies from entering final context.
- Multi-Hop Knowledge Graph: Connects policy nodes with customer segments and required documents.
- Evidence Snapshots: Supports deterministic source-level auditability for each generated answer.

## Architecture and Visual Proof

### Multi-Hop Governance Graph

This graph view demonstrates how Sentinel maps a central policy to specific customer segments and required compliance documents.


<img width="1358" height="550" alt="graph" src="https://github.com/user-attachments/assets/e04ee437-52fb-4acd-b6ff-29744b5b3327" />

### Sentinel Co-Pilot Interface

This UI demonstrates hybrid retrieval synthesizing policy rules and citing graph-backed evidence.


<img width="1854" height="806" alt="image" src="https://github.com/user-attachments/assets/1884d260-aea7-44f5-86f2-eaaa7167635d" />
<img width="1852" height="850" alt="image" src="https://github.com/user-attachments/assets/cff96d8d-20e3-48b2-aa9a-54c8dc696da3" />


## Repository Structure

```text
.
|-- api.py
|-- app.py
|-- init_graph.py
|-- prompt_modifier.py
|-- query_copilot.py
|-- seed_database.py
|-- requirements.txt
|-- frontend/
|-- img_data/
|-- submission-docs/
|-- v1_baseline_docs/
```

## Quickstart

### 1. Prerequisites

- Python 3.10+
- Neo4j Desktop (local instance on `bolt://127.0.0.1:7687`)
- API keys:
  - Groq (Llama-3)
  - Google Gemini
  - HuggingFace

### 2. Installation

```bash
git clone https://github.com/mohan120105/sentinel-graphrag.git
cd sentinel-graphrag
python -m venv venv
```

Activate virtual environment:

Windows (PowerShell):

```powershell
venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the project root:

```dotenv
NEO4J_URI=bolt://127.0.0.1:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
HUGGINGFACE_API_KEY=your_huggingface_key
```

### 4. Run the Application

```bash
streamlit run app.py
```

## Governance Query Pattern

Example guardrail pattern used in graph filtering:

```cypher
MATCH (p:Policy)
WHERE NOT ()-[:SUPERSEDES]->(p)
RETURN p
```

## Enterprise Value

- Minimizes hallucinations through layered retrieval and graph guardrails.
- Preserves compliance context and numeric fidelity from policy documents.
- Provides traceable, evidence-backed responses for regulated operations.

## Contributing

1. Fork the repository.
2. Create a feature branch.
3. Commit changes with clear messages.
4. Open a pull request.

