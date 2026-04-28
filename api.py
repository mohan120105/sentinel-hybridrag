"""Sentinel: FastAPI Control Plane for Hybrid GraphRAG and Policy Governance.

Architectural purpose:
- Exposes enterprise API routes for chat, session replay, prompt enhancement,
    and multimodal ingestion into Sentinel's governed policy graph.
- Coordinates strict retrieval generation by combining persisted conversation
    context with active-policy retrieval evidence from Neo4j.
- Provides Stateful Auditability through deterministic session/message
    persistence and citation-bearing responses.

Compliance posture:
- Enforces a Strict Retrieval Constraint pathway where answer generation is
    grounded in verified active policy context.
- Preserves Zero-Data Egress design options by integrating local edge prompt
    enhancement while keeping ingestion and retrieval flows auditable.
- Supports Multimodal Ingestion for PDF/image policy artifacts with schema
    validation prior to graph write operations.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from init_graph import GraphAction
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from neo4j import Driver, ManagedTransaction
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import BaseModel, Field, ValidationError, model_validator

from query_copilot import (
    ActivePolicy,
    STRICT_NO_ANSWER,
    build_embeddings_model,
    build_groq_llm,
    build_neo4j_driver,
    generate_answer,  # noqa: F401 – exported for external callers / testing
    load_environment,
    retrieve_active_policy,
)

try:
    from prompt_modifier import enhance_query_for_graphrag
except Exception as _pm_err:
    print(f"[WARNING] prompt_modifier not loaded: {_pm_err}")
    enhance_query_for_graphrag = None

# ── Startup ───────────────────────────────────────────────────────────────────

load_environment()

app = FastAPI(title="Sentinel Co-Pilot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-initialised singletons – created on first request to avoid blocking startup.
_driver: Driver | None = None
_llm = None
_embeddings = None


def _get_driver() -> Driver:
    """Return a cached Neo4j driver singleton.

    The driver is initialized lazily to avoid startup coupling and to preserve
    service availability during staged infrastructure bootstrapping.

    Returns:
        Driver: Reusable Neo4j driver instance.
    """
    global _driver
    if _driver is None:
        _driver = build_neo4j_driver()
    return _driver


def _get_llm():
    """Return a cached Groq LLM client singleton.

    Returns:
        ChatGroq: LLM client used for grounded response synthesis.
    """
    global _llm
    if _llm is None:
        _llm = build_groq_llm()
    return _llm


def _get_embeddings():
    """Return a cached embeddings model singleton.

    Returns:
        HuggingFaceEmbeddings: Local semantic embedding model.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = build_embeddings_model()
    return _embeddings


# ── Pydantic I/O schemas ──────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    session_id: str
    user_question: str


class EnhanceRequest(BaseModel):
    user_input: str


class EnhanceResponse(BaseModel):
    enhanced_prompt: str


class Citation(BaseModel):
    document_name: str
    category: str
    raw_score: float = Field(default=0.0)
    match_confidence: float = 0.0


class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]

    @model_validator(mode="after")
    def compute_match_confidence(self) -> "ChatResponse":
        if not self.citations:
            return self

        max_score = max(citation.raw_score for citation in self.citations)
        if max_score <= 0:
            for citation in self.citations:
                citation.match_confidence = 0.0
            return self

        for citation in self.citations:
            citation.match_confidence = round(
                (citation.raw_score / max_score) * 98.5,
                1,
            )
        return self


class UploadResponse(BaseModel):
    message: str
    document_name: str
    extracted_rule: str


class SessionSummary(BaseModel):
    session_id: str
    session_name: str
    last_timestamp: str | None = None


class SessionMessage(BaseModel):
    role: str
    content: str
    enhanced_prompt: str | None = None
    timestamp: str | None = None
    citations: List[Citation] = []


# ── Neo4j session-memory transaction functions ────────────────────────────────


def _fetch_session_history_tx(
    tx: ManagedTransaction, session_id: str
) -> List[Dict[str, str]]:
    """MERGE a session and return the last 4 messages in chronological order.

    Using a write transaction because MERGE is a write operation. The history
    is sliced in Python to keep the Cypher simple and version-agnostic.

    Args:
        tx: Active Neo4j managed transaction.
        session_id: Unique session identifier used for state continuity.

    Returns:
        List[Dict[str, str]]: Up to four role/content entries ordered
        oldest-to-newest for deterministic prompt reconstruction.
    """
    # Ensure the session node exists.
    tx.run("MERGE (s:Session {id: $session_id})", session_id=session_id)

    # Fetch the last 4 messages and return them oldest->newest for LLM injection.
    result = tx.run(
        """
        MATCH (s:Session {id: $session_id})-[:CONTAINS_MESSAGE]->(m:Message)
        WITH m
        ORDER BY m.timestamp DESC
        LIMIT 4
        RETURN m.role AS role, m.content AS content
        ORDER BY m.timestamp ASC
        """,
        session_id=session_id,
    )

    records = [
        {"role": record["role"], "content": record["content"]}
        for record in result
        if record["role"] is not None
    ]

    return records


def _save_messages_tx(
    tx: ManagedTransaction,
    session_id: str,
    user_question: str,
    enhanced_prompt: str | None,
    answer: str,
    citations: List[Citation] | None = None,
) -> None:
    """Persist user and assistant messages for a completed turn.

    The assistant timestamp is offset by 1 µs so ORDER BY timestamp ASC
    always places the user turn before the assistant turn for the same round.
    Citations are serialized to JSON for the assistant message.

    Args:
        tx: Active Neo4j managed transaction.
        session_id: Session identifier to attach Message nodes.
        user_question: Original end-user question text.
        enhanced_prompt: Optional edge-enhanced version of user text.
        answer: Assistant response content.
        citations: Optional evidence list associated with the answer.

    Returns:
        None: Data is written as a graph side effect.
    """
    now = datetime.now(timezone.utc)
    ts_user = now.isoformat()
    ts_asst = (now + timedelta(microseconds=1)).isoformat()

    citations_json = None
    if citations:
        citations_json = json.dumps([c.dict() for c in citations])

    tx.run(
        """
        MATCH (s:Session {id: $session_id})
        CREATE (u:Message {role: 'user',      content: $question, timestamp: $ts_user})
        FOREACH (_ IN CASE WHEN $enhanced_prompt IS NOT NULL THEN [1] ELSE [] END |
            SET u.enhanced_prompt = $enhanced_prompt
        )
        CREATE (a:Message {role: 'assistant', content: $answer,   timestamp: $ts_asst, citations: $citations_json})
        CREATE (s)-[:CONTAINS_MESSAGE]->(u)
        CREATE (s)-[:CONTAINS_MESSAGE]->(a)
        """,
        session_id=session_id,
        question=user_question,
        enhanced_prompt=enhanced_prompt,
        answer=answer,
        citations_json=citations_json,
        ts_user=ts_user,
        ts_asst=ts_asst,
    )


def _list_sessions_tx(tx: ManagedTransaction) -> List[Dict[str, str | None]]:
    """Return session metadata with display-name and recency semantics.

    Args:
        tx: Active Neo4j managed transaction.

    Returns:
        List[Dict[str, str | None]]: Session summaries for sidebar rendering.
    """

    result = tx.run(
        """
        MATCH (s:Session)
        OPTIONAL MATCH (s)-[:CONTAINS_MESSAGE]->(m:Message)
        WITH s, max(m.timestamp) AS last_timestamp
        OPTIONAL MATCH (s)-[:CONTAINS_MESSAGE]->(u:Message)
        WHERE u.role = 'user'
        WITH s, last_timestamp, u
        ORDER BY u.timestamp ASC
        WITH s, last_timestamp, collect(u.content)[0] AS first_user_message
        RETURN
          s.id AS session_id,
          CASE
            WHEN first_user_message IS NULL OR trim(first_user_message) = ''
              THEN 'Session ' + substring(s.id, 0, 8)
            WHEN size(first_user_message) > 48
              THEN substring(first_user_message, 0, 48) + '...'
            ELSE first_user_message
          END AS session_name,
          last_timestamp
        ORDER BY
          CASE WHEN last_timestamp IS NULL THEN 1 ELSE 0 END ASC,
          last_timestamp DESC,
          s.id ASC
        """
    )
    return [
        {
            "session_id": record["session_id"],
            "session_name": record["session_name"],
            "last_timestamp": record.get("last_timestamp"),
        }
        for record in result
    ]


def _fetch_session_messages_tx(
    tx: ManagedTransaction, session_id: str
) -> List[Dict[str, Any]]:
    """Return complete persisted message history for a session.
    
    Citations are deserialized from JSON if present.

    Args:
        tx: Active Neo4j managed transaction.
        session_id: Session identifier.

    Returns:
        List[Dict[str, Any]]: Chronological chat transcript with optional
        citations and enhanced prompt metadata.
    """

    result = tx.run(
        """
        MATCH (s:Session {id: $session_id})-[:CONTAINS_MESSAGE]->(m:Message)
        RETURN
            m.role AS role,
            m.content AS content,
            m.enhanced_prompt AS enhanced_prompt,
            m.timestamp AS timestamp,
            m.citations AS citations
        ORDER BY m.timestamp ASC
        """,
        session_id=session_id,
    )
    messages = []
    for record in result:
        if record.get("role") and record.get("content"):
            message_dict = {
                "role": record["role"],
                "content": record["content"],
                "enhanced_prompt": record.get("enhanced_prompt"),
                "timestamp": record.get("timestamp"),
                "citations": [],
            }
            
            # Recompute confidence from raw score at read time so UI evidence
            # badges remain deterministic and replayable during audits.
            citations_json = record.get("citations")
            if citations_json and isinstance(citations_json, str):
                try:
                    citations_list = json.loads(citations_json)
                    
                    # Confidence normalization is intentionally bounded to keep
                    # analyst-facing confidence labels stable across sessions.
                    if citations_list:
                        max_score = max(c.get("raw_score", 0.0) for c in citations_list if isinstance(c, dict))
                        if max_score > 0:
                            for citation in citations_list:
                                if isinstance(citation, dict):
                                    raw_score = citation.get("raw_score", 0.0)
                                    citation["match_confidence"] = round((raw_score / max_score) * 98.5, 1)
                    
                    message_dict["citations"] = citations_list
                except (json.JSONDecodeError, ValueError):
                    message_dict["citations"] = []
            
            messages.append(message_dict)
    
    return messages


# ── History-aware LLM generation ──────────────────────────────────────────────

_UPLOAD_PROMPT = (
    "Analyze this banking document. Extract the core policy rule. "
    "CRITICAL: If the document contains a table, matrix, or numerical slabs, "
    "you must include the entire table formatted as Markdown inside the "
    "'extracted_rule' string. Do not summarize away the numbers. "
    "Determine which fixed category it belongs to "
    "(Retail_Loans, Corporate_Banking, KYC_AML, Credit_Cards, Tax_Compliance). "
    "Determine if it is a CREATE_NEW policy or if it SUPERSEDE_OLD policies. "
    "Return strictly in JSON format matching the schema.\n\n"
    "Use ONLY this JSON schema with exact keys:\n"
    "{\n"
    '  "target_node": "Retail_Loans|Corporate_Banking|KYC_AML|Credit_Cards|Tax_Compliance",\n'
    '  "action_type": "CREATE_NEW|SUPERSEDE_OLD",\n'
    '  "extracted_rule": "string",\n'
    '  "superseded_document": "string or null",\n'
    '  "applies_to_customer": ["string"],\n'
    '  "requires_document": ["string"]\n'
    "}\n"
    "Do not include markdown fences. Do not include extra keys."
)

_ALLOWED_UPLOAD_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
}

_ALLOWED_UPLOAD_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}


def _normalize_graph_action_payload(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize multimodal extraction output into strict GraphAction schema.

    Args:
        raw_payload: Raw JSON object returned by the multimodal extractor.

    Returns:
        Dict[str, Any]: Schema-aligned payload suitable for strict validation.
    """

    normalized: Dict[str, Any] = dict(raw_payload)

    alias_map = {
        "category": "target_node",
        "target_category": "target_node",
        "node": "target_node",
        "action": "action_type",
        "policy_action": "action_type",
        "operation": "action_type",
        "rule": "extracted_rule",
        "policy_rule": "extracted_rule",
        "core_rule": "extracted_rule",
        "summary": "extracted_rule",
        "supersedes": "superseded_document",
        "superseded_policy": "superseded_document",
        "superseded_document_name": "superseded_document",
        "customer_types": "applies_to_customer",
        "applies_to": "applies_to_customer",
        "required_documents": "requires_document",
        "documents_required": "requires_document",
        "requires_docs": "requires_document",
    }

    for source_key, target_key in alias_map.items():
        if target_key not in normalized and source_key in normalized:
            normalized[target_key] = normalized[source_key]

    action_value = str(normalized.get("action_type", "")).strip().upper()
    if action_value in {"CREATE", "NEW", "CREATE POLICY", "CREATE_NEW_POLICY"}:
        normalized["action_type"] = "CREATE_NEW"
    elif action_value in {
        "SUPERSEDE",
        "SUPERSEDED",
        "SUPERSEDE OLD",
        "SUPERSEDE_OLD_POLICY",
    }:
        normalized["action_type"] = "SUPERSEDE_OLD"

    superseded_value = normalized.get("superseded_document")
    if isinstance(superseded_value, str) and superseded_value.strip().lower() in {
        "none",
        "null",
        "n/a",
        "na",
        "",
    }:
        normalized["superseded_document"] = None

    for list_key in ("applies_to_customer", "requires_document"):
        list_value = normalized.get(list_key)
        if list_value is None:
            normalized[list_key] = []
        elif isinstance(list_value, str):
            normalized[list_key] = [list_value.strip()] if list_value.strip() else []
        elif isinstance(list_value, list):
            normalized[list_key] = [
                str(item).strip() for item in list_value if str(item).strip()
            ]
        else:
            string_value = str(list_value).strip()
            normalized[list_key] = [string_value] if string_value else []

    return normalized


def _ingest_graph_action_to_neo4j(
    action: GraphAction,
    document_name: str,
    issue_date: str,
    source_text: str,
) -> None:
    """Insert a validated graph action into Neo4j with lineage semantics.

    Args:
        action: Strictly validated graph mutation payload.
        document_name: Source document identifier used as policy node name.
        issue_date: Effective issue date for the ingested policy record.
        source_text: Ingestion provenance text retained for traceability.

    Returns:
        None: Writes policy, relation, and optional supersession edges.
    """

    create_policy_query = """
    MATCH (c:Category {name: $category_name})
    CREATE (p:Policy {
        name: $policy_name,
        issue_date: $issue_date,
        source_text: $source_text,
        extracted_rule: $extracted_rule,
        embedding: $embedding,
        action_type: $action_type,
        active: true,
        created_at: datetime($created_at)
    })
    MERGE (p)-[:BELONGS_TO]->(c)
    WITH p
    UNWIND (CASE WHEN size($applies_to_customer) > 0 THEN $applies_to_customer ELSE [null] END) AS customer
    FOREACH (ignoreMe IN CASE WHEN customer IS NOT NULL THEN [1] ELSE [] END |
        MERGE (ct:CustomerType {name: customer})
        MERGE (p)-[:APPLIES_TO]->(ct)
    )
    WITH p
    UNWIND (CASE WHEN size($requires_document) > 0 THEN $requires_document ELSE [null] END) AS doc
    FOREACH (ignoreMe IN CASE WHEN doc IS NOT NULL THEN [1] ELSE [] END |
        MERGE (dr:DocumentRequirement {name: doc})
        MERGE (p)-[:REQUIRES]->(dr)
    )
    RETURN p.name AS created_policy
    """

    supersede_query = """
    MATCH (new_policy:Policy {name: $new_policy_name})
    MATCH (old_policy:Policy {name: $old_policy_name})
    MERGE (new_policy)-[:SUPERSEDES]->(old_policy)
    SET old_policy.active = false,
        old_policy.retired_at = datetime($retired_at)
    RETURN new_policy.name AS new_policy_name, old_policy.name AS old_policy_name
    """

    embeddings_model = _get_embeddings()
    # Embedding full rule plus provenance text improves downstream retrieval
    # recall while preserving evidence lineage for compliance review.
    semantic_text = f"{action.extracted_rule}\n\n{source_text}"
    embedding = [float(value) for value in embeddings_model.embed_query(semantic_text)]
    timestamp = datetime.utcnow().isoformat()

    with _get_driver().session() as session:
        session.execute_write(
            lambda tx: tx.run(
                create_policy_query,
                category_name=action.target_node,
                policy_name=document_name,
                issue_date=issue_date,
                source_text=source_text,
                extracted_rule=action.extracted_rule,
                embedding=embedding,
                action_type=action.action_type,
                applies_to_customer=action.applies_to_customer,
                requires_document=action.requires_document,
                created_at=timestamp,
            ).consume()
        )

        if action.action_type == "SUPERSEDE_OLD" and action.superseded_document:
            session.execute_write(
                lambda tx: tx.run(
                    supersede_query,
                    new_policy_name=document_name,
                    old_policy_name=action.superseded_document,
                    retired_at=timestamp,
                ).consume()
            )


def _extract_graph_action_from_upload(
    file_bytes: bytes,
    filename: str,
    mime_type: str,
) -> GraphAction:
    """Extract and validate a GraphAction from uploaded multimodal content.

    Args:
        file_bytes: Binary payload of the uploaded file.
        filename: Original filename for error context and lineage metadata.
        mime_type: Effective MIME type validated against allow-list.

    Returns:
        GraphAction: Strictly validated graph action ready for ingestion.

    Raises:
        ValueError: If API configuration is missing or extraction output is
            empty/invalid.
    """

    from google import genai
    from google.genai import types as genai_types

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is missing in environment configuration.")

    client = genai.Client(api_key=gemini_api_key)
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_bytes(
                        data=file_bytes,
                        mime_type=mime_type,
                    ),
                    genai_types.Part.from_text(text=_UPLOAD_PROMPT),
                ],
            )
        ],
        config=genai_types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    response_text = (response.text or "").strip()
    if not response_text:
        raise ValueError(f"Gemini returned an empty extraction for {filename}.")

    if response_text.startswith("```"):
        response_text = response_text.strip("`")
        response_text = response_text.replace("json", "", 1).strip()

    try:
        import json

        payload = json.loads(response_text)
    except Exception as exc:
        raise ValueError(f"Gemini returned invalid JSON for {filename}: {exc}") from exc

    normalized_payload = _normalize_graph_action_payload(payload)
    return GraphAction.model_validate(normalized_payload)


def _validate_upload_type(upload: UploadFile) -> str:
    """Validate upload content type/extension against ingestion allow-list.

    Args:
        upload: Uploaded file object from FastAPI request binding.

    Returns:
        str: Normalized MIME type used by multimodal extraction.

    Raises:
        HTTPException: If the file type is outside approved formats.
    """

    filename = upload.filename or ""
    extension = os.path.splitext(filename)[1].lower()
    content_type = (upload.content_type or "application/octet-stream").lower()

    if content_type in _ALLOWED_UPLOAD_TYPES:
        return content_type
    if extension in _ALLOWED_UPLOAD_EXTENSIONS:
        if extension == ".pdf":
            return "application/pdf"
        if extension == ".png":
            return "image/png"
        return "image/jpeg"

    raise HTTPException(
        status_code=400,
        detail="Only PDF, PNG, JPG, and JPEG uploads are supported.",
    )


def _generate_with_history(
    llm,
    active_context: List[ActivePolicy],
    user_question: str,
    history: List[Dict[str, str]],
) -> str:
    """Generate a grounded answer using active policies and session history.

    Falls back to STRICT_NO_ANSWER when active_context is empty, consistent
    with the base generate_answer behaviour in query_copilot.py.

    Args:
        llm: LLM client used for final response generation.
        active_context: Retrieved active policy evidence set.
        user_question: Current user turn.
        history: Prior role/content messages for continuity.

    Returns:
        str: Grounded assistant response or strict fallback string.
    """
    if not active_context:
        return STRICT_NO_ANSWER

    context_blocks = [
        (
            f"Document: {p.document_name}\n"
            f"Category: {p.category}\n"
            f"Applies To: {', '.join(p.customer_types) if p.customer_types else 'None'}\n"
            f"Requires: {', '.join(p.required_docs) if p.required_docs else 'None'}\n"
            f"Rule: {p.extracted_rule}"
        )
        for p in active_context
    ]
    context_text = "\n\n".join(context_blocks)

    history_messages = [
        {
            "role": item["role"],
            "content": item["content"],
        }
        for item in history
        if item.get("role") in {"user", "assistant"} and item.get("content")
    ]

    system_content = (
        # This system directive encodes strict grounding policy to reduce
        # unsupported synthesis and strengthen retrieval-compliance guarantees.
        "You are the Sentinel Banking Co-Pilot. "
        "Answer the user's question using ONLY the provided active_context. "
        "If the context is empty or does not contain the answer, you MUST reply with: "
        '"I cannot find a verified active policy for this in the current database."\n\n'
        f"active_context:\n{context_text}"
    )

    llm_messages = [SystemMessage(content=system_content)]
    for msg in history_messages:
        if msg["role"] == "assistant":
            llm_messages.append(AIMessage(content=msg["content"]))
        else:
            llm_messages.append(HumanMessage(content=msg["content"]))

    llm_messages.append(HumanMessage(content=user_question))

    try:
        response = llm.invoke(llm_messages)
        return str(response.content).strip()
    except Exception as exc:
        err = str(exc)
        if "429" in err or "rate" in err.lower():
            return (
                "Groq API rate limit encountered while generating response. "
                "Please retry in a few seconds."
            )
        return f"Failed to generate response from Groq: {exc}"


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/sessions", response_model=List[SessionSummary])
def list_sessions() -> List[SessionSummary]:
    """List all persisted sessions with display-friendly metadata.

    Returns:
        List[SessionSummary]: Session identifiers, names, and recency stamps.

    Raises:
        HTTPException: If Neo4j cannot be reached.
    """

    try:
        with _get_driver().session() as neo4j_session:
            records = neo4j_session.execute_read(_list_sessions_tx)
            return [SessionSummary(**record) for record in records]
    except (Neo4jError, ServiceUnavailable) as exc:
        raise HTTPException(
            status_code=503, detail=f"Neo4j unavailable: {exc}"
        ) from exc


@app.get("/sessions/{session_id}/messages", response_model=List[SessionMessage])
def get_session_messages(session_id: str) -> List[SessionMessage]:
    """Return full persisted chat history for one session.

    Args:
        session_id: Session identifier from route path.

    Returns:
        List[SessionMessage]: Chronological transcript payload.

    Raises:
        HTTPException: If validation fails or Neo4j is unavailable.
    """

    trimmed_session_id = session_id.strip()
    if not trimmed_session_id:
        raise HTTPException(status_code=422, detail="session_id must not be empty.")

    try:
        with _get_driver().session() as neo4j_session:
            records = neo4j_session.execute_read(
                _fetch_session_messages_tx, trimmed_session_id
            )
            return [SessionMessage(**record) for record in records]
    except (Neo4jError, ServiceUnavailable) as exc:
        raise HTTPException(
            status_code=503, detail=f"Neo4j unavailable: {exc}"
        ) from exc


@app.post("/enhance", response_model=EnhanceResponse)
def enhance_prompt(request: EnhanceRequest) -> EnhanceResponse:
    """Enhance raw user text using local edge prompt optimization.

    Args:
        request: API payload containing user input text.

    Returns:
        EnhanceResponse: Enhanced query text for higher-fidelity retrieval.

    Raises:
        HTTPException: If input is invalid or local model is unavailable.
    """

    user_input = request.user_input.strip()
    if not user_input:
        raise HTTPException(status_code=422, detail="user_input must not be empty.")

    if enhance_query_for_graphrag is None:
        raise HTTPException(
            status_code=503,
            detail="Local prompt enhancement model is unavailable on this server.",
        )

    try:
        enhanced_prompt = enhance_query_for_graphrag(user_input).strip()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local prompt enhancement failed: {exc}",
        ) from exc

    return EnhanceResponse(enhanced_prompt=enhanced_prompt or user_input)


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """GraphRAG chat endpoint with persistent Neo4j session memory.

    Flow:
    1. MERGE Session node + fetch last 4 messages as conversation history.
    2. Embed the question; run True Hybrid GraphRAG to retrieve active policies.
    3. Generate LLM answer using active_context + conversation history.
    4. Persist new user + assistant Message nodes to the graph.
    5. Return the answer and extracted citations to the client.

    Args:
        request: Chat request containing session identifier and user question.

    Returns:
        ChatResponse: Grounded answer plus citation set for client evidence UI.

    Raises:
        HTTPException: If request validation fails or session operations cannot
            access Neo4j.
    """
    session_id = request.session_id.strip()
    user_question = request.user_question.strip()

    if not session_id:
        raise HTTPException(status_code=422, detail="session_id must not be empty.")
    if not user_question:
        raise HTTPException(status_code=422, detail="user_question must not be empty.")

    driver = _get_driver()
    llm = _get_llm()
    embeddings = _get_embeddings()

    # ── Step 1: Fetch (or create) session + last 4 message history ────────────
    try:
        with driver.session() as neo4j_session:
            history = neo4j_session.execute_write(
                _fetch_session_history_tx, session_id
            )
    except (Neo4jError, ServiceUnavailable) as exc:
        raise HTTPException(
            status_code=503, detail=f"Neo4j session error: {exc}"
        ) from exc

    # Step 2 uses true hybrid retrieval, mathematically fusing Lucene BM25
    # keyword relevance with cosine-similarity vector search before governance
    # filtering for active policy truth.
    question_embedding = embeddings.embed_query(user_question)
    active_context = retrieve_active_policy(driver, user_question, question_embedding)

    # ── Step 3: Generate answer with retrieved context + conversation history ──
    answer = _generate_with_history(llm, active_context, user_question, history)

    # ── Step 4: Build citation list for the client ─────────────────────────────
    citations = [
        Citation(
            document_name=p.document_name,
            category=p.category,
            raw_score=float(p.score),
        )
        for p in active_context
    ]

    # ── Step 5: Persist the new turn to Neo4j with citations (non-fatal on failure) ──────────
    try:
        with driver.session() as neo4j_session:
            neo4j_session.execute_write(
                _save_messages_tx,
                session_id,
                user_question,
                None,
                answer,
                citations,
            )
    except (Neo4jError, ServiceUnavailable) as exc:
        # Answer was already generated; log and continue rather than raising.
        print(
            f"[WARNING] Could not persist messages for session '{session_id}': {exc}"
        )

    # ── Step 6: Return response with answer and citations ──────────────────────────
    return ChatResponse(answer=answer, citations=citations)


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Ingest uploaded policy artifacts via multimodal extraction and graph write.

    Args:
        file: Uploaded PDF/image from the client ingestion panel.

    Returns:
        UploadResponse: Ingestion status with document lineage details.

    Raises:
        HTTPException: If validation, extraction, schema, or persistence fails.
    """

    filename = (file.filename or "uploaded_document").strip()
    if not filename:
        raise HTTPException(status_code=422, detail="Uploaded file must have a name.")

    mime_type = _validate_upload_type(file)

    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=422, detail="Uploaded file is empty.")

        action = _extract_graph_action_from_upload(file_bytes, filename, mime_type)
        # Multimodal Ingestion result is projected into graph-native policy
        # entities to keep downstream retrieval traceable and governance-ready.
        inferred_source_text = (
            f"Multimodal ingestion from {filename}. "
            f"Rule: {action.extracted_rule}"
        )
        _ingest_graph_action_to_neo4j(
            action=action,
            document_name=filename,
            issue_date=datetime.utcnow().date().isoformat(),
            source_text=inferred_source_text,
        )

        return UploadResponse(
            message="Document ingested successfully.",
            document_name=filename,
            extracted_rule=action.extracted_rule,
        )
    except HTTPException:
        raise
    except ValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Schema validation failed during ingestion: {exc}",
        ) from exc
    except (Neo4jError, ServiceUnavailable) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Neo4j ingestion failed: {exc}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {exc}",
        ) from exc
