"""Sentinel: Retrieval and Grounded Generation Core for Active Policy Q&A."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from neo4j import Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable
import requests
from pydantic import BaseModel, Field

from connect import build_neo4j_driver as _build_neo4j_driver_from_connect

# Default model lookup paths (env override supported)
FASTTEXT_MODEL_ENV = os.getenv("FASTTEXT_LANG_MODEL")
DEFAULT_FASTTEXT_PATHS = [
    FASTTEXT_MODEL_ENV,
    os.path.join(os.path.dirname(__file__), "models", "lid.176.bin"),
    os.path.join(os.path.dirname(__file__), "lid.176.bin"),
]

# Minimal ISO code -> language name mapping
LANG_CODE_TO_NAME = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "zh": "Chinese",
    "ar": "Arabic",
    "bn": "Bengali",
    "pa": "Punjabi",
    "mr": "Marathi",
    "ta": "Tamil",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "ur": "Urdu",
}

_FASTTEXT_MODEL: Any = None


def _find_fasttext_model() -> str | None:
    """Return path to FastText model if present, else None."""

    candidates: list[str] = []
    for model_path in DEFAULT_FASTTEXT_PATHS:
        if not model_path:
            continue
        candidates.append(model_path)
        root, ext = os.path.splitext(model_path)
        if not ext:
            candidates.append(root + ".bin")
            candidates.append(root + ".ftz")
        elif ext.lower() == ".bin":
            candidates.append(root + ".ftz")
        elif ext.lower() == ".ftz":
            candidates.append(root + ".bin")

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def ensure_fasttext_model() -> str | None:
    """Load FastText model if available and return its resolved path."""

    global _FASTTEXT_MODEL
    if _FASTTEXT_MODEL is not None:
        return _find_fasttext_model()

    model_path = _find_fasttext_model()
    if not model_path:
        return None

    try:
        import fasttext as _fasttext  # type: ignore

        _FASTTEXT_MODEL = _fasttext.load_model(model_path)  # type: ignore
        return model_path
    except Exception:
        _FASTTEXT_MODEL = None
        return None



STRICT_NO_ANSWER = (
    "I cannot find a verified active policy for this in the current database."
)

# Embedding dimensionality expected from the multilingual MiniLM paraphrase model (384)
EMBEDDING_DIM = 384


class ActivePolicy(BaseModel):
    """Verified active policy context returned from Neo4j retrieval."""

    document_name: str = Field(..., description="Policy document identifier.")
    category: str = Field(..., description="SME-governed ontology category.")
    customer_types: List[str] = Field(
        default_factory=list,
        description="Customer types explicitly connected through APPLIES_TO edges.",
    )
    required_docs: List[str] = Field(
        default_factory=list,
        description="Required documents explicitly connected through REQUIRES edges.",
    )
    extracted_rule: str = Field(..., description="Normalized policy rule summary.")
    source_text: str = Field(..., description="Original policy source text.")
    score: float = Field(..., description="Raw hybrid retrieval score from Neo4j.")
    match_confidence: float = Field(
        ...,
        description="Normalized retrieval confidence percentage for UI display.",
    )
    version_status: str = Field(
        ...,
        description="Version status flag for response metadata.",
    )


def _normalize_match_confidence(score: float, max_score: float) -> float:
    """Normalize raw retrieval scores for stable UI confidence display.

    Args:
        score: Raw score for a candidate policy.
        max_score: Highest score in the retrieved result set.

    Returns:
        float: Bounded confidence value scaled for analyst readability.
    """

    if max_score <= 0:
        return 0.0
    normalized = max(0.0, min(score / max_score, 1.0))
    return round(normalized * 96.5, 1)


def detect_user_language(text: str) -> str:
    """Detect user language using FastText with langdetect fallback.

    Returns a full language name (e.g., 'Telugu', 'Hindi', 'Spanish').
    Defaults to 'English' on low confidence or errors.
    """
    text = (text or "").strip()
    if not text:
        return "English"

    # 1) FastText path
    try:
        model_path = ensure_fasttext_model()
        if model_path and _FASTTEXT_MODEL is not None:
            labels, probs = _FASTTEXT_MODEL.predict(text, k=1)  # type: ignore
            if labels and probs:
                code = labels[0].replace("__label__", "")
                confidence = float(probs[0])
                if confidence >= 0.50:
                    return LANG_CODE_TO_NAME.get(code, code)
                # low confidence -> fall through to fallback
    except Exception:
        pass

    # 2) langdetect fallback
    try:
        from langdetect import detect as _langdetect_detect  # type: ignore

        if _langdetect_detect is not None:
            code = _langdetect_detect(text)  # type: ignore
            return LANG_CODE_TO_NAME.get(code, code)
    except Exception:
        pass

    # Default
    return "English"


def load_environment() -> None:
    """Load environment variables and sanitize credential formatting.

    Returns:
        None: Environment is updated in process memory.
    """

    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path=dotenv_path, override=True)

    # Normalize values to avoid quoted strings breaking auth/uri parsing.
    for key in (
        "GROQ_API_KEY",
        "NEO4J_URI",
        "NEO4J_USER",
        "NEO4J_USERNAME",
        "NEO4J_PASSWORD",
    ):
        value = os.environ.get(key)
        if value is not None:
            os.environ[key] = value.strip().strip('"').strip("'")

    neo4j_user = os.environ.get("NEO4J_USER")
    neo4j_username = os.environ.get("NEO4J_USERNAME")
    if neo4j_user and not neo4j_username:
        os.environ["NEO4J_USERNAME"] = neo4j_user
    elif neo4j_username and not neo4j_user:
        os.environ["NEO4J_USER"] = neo4j_username


def _load_and_sanitize_env() -> None:
    """Backward-compatible alias for legacy callers.

    Returns:
        None: Delegates to load_environment.
    """

    load_environment()


def _to_bolt_uri(uri: str) -> str:
    """Convert Neo4j routing URI formats into direct Bolt transport forms.

    Args:
        uri: Original configured Neo4j URI.

    Returns:
        str: Direct URI variant suitable for non-clustered deployments.
    """

    if uri.startswith("neo4j://"):
        return uri.replace("neo4j://", "bolt://", 1)
    if uri.startswith("neo4j+s://"):
        return uri.replace("neo4j+s://", "bolt+s://", 1)
    if uri.startswith("neo4j+ssc://"):
        return uri.replace("neo4j+ssc://", "bolt+ssc://", 1)
    return uri



def build_neo4j_driver() -> Driver:
    """Create Neo4j driver from the shared environment-driven factory.

    Returns:
        Driver: Verified Neo4j driver instance.
    """

    return _build_neo4j_driver_from_connect()


def build_groq_llm() -> ChatGroq:
    """Build Groq LLM client used for grounded response synthesis.

    Returns:
        ChatGroq: Configured deterministic LLM client.

    Raises:
        ValueError: If GROQ_API_KEY is absent.
    """

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Export it before running this script.")

    # Ensure langchain global compatibility (some versions expect langchain.verbose)
    try:
        import langchain as _langchain

        if not hasattr(_langchain, "verbose"):
            setattr(_langchain, "verbose", False)
    except Exception:
        pass

    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)


def build_embeddings_model():
    """Build a Gradio Space-backed embeddings client.

    The returned object exposes `embed_query(text)` so the rest of the
    retrieval pipeline can stay unchanged.
    """

    from gradio_client import Client

    space_name = os.getenv("HF_EMBEDDING_SPACE", "mohan1201/sentinel-embedding-server")
    client = Client(space_name)

    # Default to 'embed' endpoint; override via HF_EMBEDDING_API_NAME env var if needed
    api_name = os.getenv("HF_EMBEDDING_API_NAME", "embed").lstrip('/')

    def get_embedding(text: str):
        return client.predict(text, api_name=api_name)

    class _GradioSpaceEmbeddings:
        def embed_query(self, text: str):
            result = get_embedding(text)
            if isinstance(result, list):
                if result and isinstance(result[0], (int, float)):
                    return result
                if result and isinstance(result[0], list):
                    return result[0]
            return result

    return _GradioSpaceEmbeddings()


def retrieve_active_policy(
    driver: Driver,
    user_question: str,
    question_embedding: Sequence[float],
    top_k: int = 5,
    only_latest: bool = True,
    user_tier: int = 1,
    similarity_threshold: float = 0.75,
) -> List[ActivePolicy]:
    """Retrieve active policies with vector search and governance filtering.

    Business logic:
    - Semantic candidate generation via Neo4j vector index on Policy.embedding.
        - Strict governance filter excludes superseded nodes without hard-binding
            the SUPERSEDES relationship type in the pattern.
    - Returns evidence-ready context (rule, document name, category).

    Args:
        driver: Active Neo4j driver.
        user_question: Original user question text.
        question_embedding: Dense embedding vector for semantic lookup.
        top_k: Maximum number of candidate policies to return.
        only_latest: When True, exclude policies superseded by newer versions.

    Returns:
        List[ActivePolicy]: Ranked active-policy evidence records.
    """

    cypher_query = """
    CALL {
        WITH $question_embedding AS qe
        MATCH (p:Policy)
        SEARCH p IN (
            VECTOR INDEX policy_embeddings
            FOR qe
            LIMIT $top_k
        ) SCORE AS vector_score
        WHERE ($user_tier >= p.access_code) AND vector_score > $similarity_threshold
        RETURN p, vector_score, 0.0 AS text_score

        UNION ALL

        WITH $user_question AS uq, $top_k AS tk
        CALL db.index.fulltext.queryNodes('policy_keywords', uq, {limit: tk})
        YIELD node AS p, score AS raw_text_score
        WHERE ($user_tier >= p.access_code)
        RETURN p, 0.0 AS vector_score, raw_text_score AS text_score
    }
    // 1. Score Fusion
    WITH p, max(vector_score) AS vs, max(text_score) AS ts
    // Normalize BM25 score and mathematically fuse it with cosine-style
    // vector similarity to achieve true hybrid retrieval behavior.
    WITH p, (vs + (ts / 10.0)) AS combined_score
    WHERE vs > $similarity_threshold

    // 2. Governance Firewall
    // Optionally exclude superseded nodes so only latest policy truth flows
    // into generation and downstream compliance decisions.
    OPTIONAL MATCH (superseder)-[supersedes_rel]->(p)
    WHERE type(supersedes_rel) = 'SUPERSEDES'
    WITH p, combined_score, count(supersedes_rel) AS supersedes_count, $only_latest AS only_latest
    WHERE (NOT only_latest) OR supersedes_count = 0

    OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)

    // 3. Multi-Hop Extraction
    OPTIONAL MATCH (p)-[:APPLIES_TO]->(ct:CustomerType)
    OPTIONAL MATCH (p)-[:REQUIRES]->(dr:DocumentRequirement)
    WITH p, c, combined_score, supersedes_count, collect(DISTINCT ct.name) AS customer_types, collect(DISTINCT dr.name) AS required_docs
    RETURN p.name AS document_name,
           coalesce(c.name, "General") AS category,
           coalesce(p.extracted_rule, "") AS extracted_rule,
           coalesce(p.source_text, "") AS source_text,
           customer_types,
           required_docs,
            combined_score AS score,
            CASE WHEN supersedes_count = 0 THEN "LATEST" ELSE "SUPERSEDED" END AS version_status
    ORDER BY score DESC
    LIMIT $top_k
    """

    vector_only_query = """
    MATCH (p:Policy)
    SEARCH p IN (
        VECTOR INDEX policy_embeddings
        FOR $question_embedding
        LIMIT $top_k
    ) SCORE AS score
    WHERE ($user_tier >= p.access_code) AND score > $similarity_threshold
    OPTIONAL MATCH (superseder)-[supersedes_rel]->(p)
    WHERE type(supersedes_rel) = 'SUPERSEDES'
    WITH p, score, count(supersedes_rel) AS supersedes_count, $only_latest AS only_latest
    WHERE (NOT only_latest) OR supersedes_count = 0

    OPTIONAL MATCH (p)-[:BELONGS_TO]->(c:Category)
    OPTIONAL MATCH (p)-[:APPLIES_TO]->(ct:CustomerType)
    OPTIONAL MATCH (p)-[:REQUIRES]->(dr:DocumentRequirement)
    WITH p, c, score, supersedes_count, collect(DISTINCT ct.name) AS customer_types, collect(DISTINCT dr.name) AS required_docs
    RETURN p.name AS document_name,
           coalesce(c.name, "General") AS category,
           coalesce(p.extracted_rule, "") AS extracted_rule,
           coalesce(p.source_text, "") AS source_text,
           customer_types,
           required_docs,
            score,
            CASE WHEN supersedes_count = 0 THEN "LATEST" ELSE "SUPERSEDED" END AS version_status
    ORDER BY score DESC
    LIMIT $top_k
    """

    def _is_missing_fulltext_index(error: Neo4jError) -> bool:
        """Detect missing full-text index errors for graceful fallback.

        Args:
            error: Neo4j exception from hybrid query execution.

        Returns:
            bool: True when the policy keyword index is unavailable.
        """

        error_text = str(error).lower()
        return (
            "policy_keywords" in error_text
            and "index" in error_text
            and (
                "does not exist" in error_text
                or "not found" in error_text
                or "unknown" in error_text
                or "there is no such" in error_text
            )
        )

    try:
        with driver.session() as session:
            query_params = {
                "user_question": user_question,
                "question_embedding": [float(value) for value in question_embedding],
                "top_k": top_k,
                "only_latest": only_latest,
                "user_tier": user_tier,
                "similarity_threshold": float(similarity_threshold),
            }
            try:
                records = session.execute_read(
                    lambda tx: list(
                        tx.run(
                            cypher_query,
                            **query_params,
                        )
                    )
                )
            except Neo4jError as error:
                if not _is_missing_fulltext_index(error):
                    raise
                print(
                    "Full-text index 'policy_keywords' is unavailable; "
                    "falling back to vector-only retrieval."
                )
                records = session.execute_read(
                    lambda tx: list(
                        tx.run(
                            vector_only_query,
                            question_embedding=query_params["question_embedding"],
                            top_k=top_k,
                            only_latest=only_latest,
                            user_tier=user_tier,
                            similarity_threshold=query_params["similarity_threshold"],
                        )
                    )
                )

        if not records:
            return []

        # Optional: filter by sentence-similarity endpoint if configured.
        hf_sim_endpoint = os.getenv("HF_SIMILARITY_ENDPOINT")
        hf_token = os.getenv("HF_TOKEN")
        if hf_sim_endpoint and hf_token:
            try:
                sentences = [str(rec.get("source_text", "")) for rec in records]
                payload = {"inputs": {"source_sentence": f"{user_question}", "sentences": sentences}}
                headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
                resp = requests.post(hf_sim_endpoint, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                sim_results = resp.json()
                filtered = []
                for rec, score in zip(records, sim_results if isinstance(sim_results, list) else []):
                    try:
                        s = float(score)
                    except Exception:
                        if isinstance(score, dict) and "score" in score:
                            s = float(score["score"])
                        else:
                            s = 0.0
                    if s >= float(similarity_threshold):
                        filtered.append(rec)
                records = filtered
            except Exception as _sim_err:
                print(f"Similarity endpoint filtering failed: {_sim_err}")

        max_score = max(float(record["score"]) for record in records) if records else 0.0

        policies: List[ActivePolicy] = []

        for record in records:
            raw_score = float(record["score"])
            policies.append(
                ActivePolicy(
                    document_name=record["document_name"],
                    category=record["category"],
                    customer_types=[
                        value
                        for value in (record.get("customer_types") or [])
                        if value is not None
                    ],
                    required_docs=[
                        value
                        for value in (record.get("required_docs") or [])
                        if value is not None
                    ],
                    extracted_rule=record["extracted_rule"],
                    source_text=record["source_text"],
                    score=raw_score,
                    match_confidence=_normalize_match_confidence(raw_score, max_score),
                    version_status=record["version_status"],
                )
            )

        return policies
    except ServiceUnavailable as error:
        print(f"Neo4j connection dropped during retrieval: {error}")
        return []
    except Neo4jError as error:
        print(f"Neo4j query error during retrieval: {error}")
        return []
    except Exception as error:
        print(f"Unexpected retrieval error: {error}")
        return []


def generate_answer(
    llm: ChatGroq,
    active_context: Sequence[ActivePolicy],
    user_question: str,
    detected_language: str = "English",
) -> str:
    """Generate grounded response text from verified active policy context.

    Args:
        llm: LLM client used for final answer generation.
        active_context: Active policy evidence retrieved from Neo4j.
        user_question: User's question text.

    Returns:
        str: Grounded answer text or strict no-answer fallback.
    """

    if not active_context:
        return STRICT_NO_ANSWER

    context_blocks = []
    for item in active_context:
        context_blocks.append(
            (
                f"Document: {item.document_name}\n"
                f"Category: {item.category}\n"
                f"Applies To: {', '.join(item.customer_types) if item.customer_types else 'None'}\n"
                f"Requires: {', '.join(item.required_docs) if item.required_docs else 'None'}\n"
                f"Rule: {item.extracted_rule}"
            )
        )
    context_text = "\n\n".join(context_blocks)

    prompt = PromptTemplate.from_template(
        """
The user's query is in {detected_language}. You MUST answer only in
{detected_language}; do not switch languages mid-response.
Use the provided English context to generate a precise, Fact-Strict compliance
response in {detected_language}.
You MUST include specific numbers (e.g., 10%, 20% TDS rates) and Document IDs
(e.g., AUDIT-2026-Q1-RED) found in the context. Keep technical acronyms like
'TDS' and 'KYC' in English for regulatory clarity.

You are a Strict Compliance Auditor. Your job is to extract precise, verifiable
regulatory facts from the provided English `active_context` and present them in
{detected_language}. Focus specifically on numeric policy values such as TDS
rates, liquidity ratios/percentages, thresholds, fines, and durations. Use
ONLY the provided `active_context` (which is in English) to derive facts; do
NOT hallucinate, estimate, or invent values.

If the context does not contain the requested fact, reply exactly:
"I cannot find a verified active policy for this in the current database."

When returning numbers, include the source document name/ID for each extracted
fact and preserve the original numeric formatting (e.g., "8.35%", "INR 50,000").

active_context:
{active_context}

user_question:
{user_question}
""".strip()
    )

    try:
        formatted_prompt = prompt.format(
            active_context=context_text,
            user_question=user_question,
            detected_language=detected_language,
        )
        response = llm.invoke(formatted_prompt)
        return str(response.content).strip()
    except Exception as error:
        error_text = str(error)
        if "429" in error_text or "rate" in error_text.lower():
            return (
                "Groq API rate limit encountered while generating response. "
                "Please retry in a few seconds."
            )
        return f"Failed to generate response from Groq: {error}"


def print_response(answer: str, active_context: Sequence[ActivePolicy]) -> None:
    """Print answer text along with evidence snapshot for operators.

    Args:
        answer: Final generated answer.
        active_context: Retrieved evidence records used for grounding.

    Returns:
        None: Writes formatted output to console.
    """

    if active_context:
        evidence = ", ".join(
            f"{item.document_name} [{item.category}] (score={item.score:.4f})"
            for item in active_context
        )
    else:
        evidence = "None"

    print("\nAnswer:")
    print(answer)
    print(f"Source: {evidence}\n")


def main() -> None:
    """Run interactive CLI loop for Sentinel retrieval and generation.

    Returns:
        None: Runs until explicit user exit.
    """

    load_environment()

    try:
        driver = build_neo4j_driver()
    except ServiceUnavailable as error:
        print("Neo4j is not reachable. Start Neo4j and confirm Bolt is enabled.")
        expected_endpoint = os.getenv("NEO4J_URI") or "<unset>"
        print(f"Expected endpoint: {expected_endpoint}")
        print(f"Connection error: {error}")
        return
    except Neo4jError as error:
        print(f"Neo4j startup check failed: {error}")
        return

    try:
        llm = build_groq_llm()
        embeddings_model = build_embeddings_model()
        print("Sentinel Co-Pilot is ready. Type 'exit' to quit.")

        while True:
            user_question = input("\nAsk Sentinel> ").strip()
            if user_question.lower() in {"exit", "quit", "q"}:
                print("Exiting Sentinel Co-Pilot.")
                break

            # Detect user's language (FastText preferred, langdetect fallback)
            detected_language = detect_user_language(user_question)

            question_embedding = embeddings_model.embed_query(f"query: {user_question}")

            active_context = retrieve_active_policy(
                driver,
                user_question,
                question_embedding,
                top_k=5,
                similarity_threshold=0.75,
            )
            answer = generate_answer(llm, active_context, user_question, detected_language)
            print_response(answer, active_context)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
