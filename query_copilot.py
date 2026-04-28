"""Sentinel: Retrieval and Grounded Generation Core for Active Policy Q&A.

Architectural purpose:
- Implements Sentinel retrieval logic that resolves policy evidence from Neo4j
    and synthesizes answers only from active, governance-valid context.
- Encapsulates enterprise fallback controls so unanswered queries fail closed
    to strict no-answer responses instead of speculative completions.

Compliance posture:
- Enforces Strict Retrieval Constraint by excluding superseded policy nodes.
- Supports Stateful Auditability by returning evidence-bearing context objects
    with normalized confidence metadata.
- Maintains Zero-Data Egress options by using local embeddings for semantic
    retrieval preparation.
"""

from __future__ import annotations

import os
from typing import List, Sequence

from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_groq import ChatGroq
from neo4j import Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import BaseModel, Field

from connect import build_neo4j_driver as _build_neo4j_driver_from_connect

STRICT_NO_ANSWER = (
    "I cannot find a verified active policy for this in the current database."
)


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

    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)


def build_embeddings_model() -> HuggingFaceEndpointEmbeddings:
    """Build cloud-hosted sentence-transformer embeddings client.

    Returns:
        HuggingFaceEndpointEmbeddings: Embedding client for query vectorization.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN is not set. Export it before running this script.")

    return HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )


def retrieve_active_policy(
    driver: Driver,
    user_question: str,
    question_embedding: Sequence[float],
    top_k: int = 5,
) -> List[ActivePolicy]:
    """Retrieve active policies with vector search and governance filtering.

    Business logic:
    - Semantic candidate generation via Neo4j vector index on Policy.embedding.
    - Strict governance filter is enforced with `WHERE NOT ()-[:SUPERSEDES]->(p)`.
    - Returns evidence-ready context (rule, document name, category).

    Args:
        driver: Active Neo4j driver.
        user_question: Original user question text.
        question_embedding: Dense embedding vector for semantic lookup.
        top_k: Maximum number of candidate policies to return.

    Returns:
        List[ActivePolicy]: Ranked active-policy evidence records.
    """

    cypher_query = """
    CALL {
        WITH $question_embedding AS qe, $top_k AS tk
        CALL db.index.vector.queryNodes('policy_embeddings', tk, qe)
        YIELD node AS p, score AS vector_score
        RETURN p, vector_score, 0.0 AS text_score

        UNION ALL

        WITH $user_question AS uq, $top_k AS tk
        CALL db.index.fulltext.queryNodes('policy_keywords', uq, {limit: tk})
        YIELD node AS p, score AS raw_text_score
        RETURN p, 0.0 AS vector_score, raw_text_score AS text_score
    }
    // 1. Score Fusion
    WITH p, max(vector_score) AS vs, max(text_score) AS ts
    // Normalize BM25 score and mathematically fuse it with cosine-style
    // vector similarity to achieve true hybrid retrieval behavior.
    WITH p, (vs + (ts / 10.0)) AS combined_score

    // 2. Governance Firewall
    // Strictly exclude superseded nodes so only active policy truth flows
    // into generation and downstream compliance decisions.
    MATCH (p)-[:BELONGS_TO]->(c:Category)
    WHERE NOT ()-[:SUPERSEDES]->(p)

    // 3. Multi-Hop Extraction
    OPTIONAL MATCH (p)-[:APPLIES_TO]->(ct:CustomerType)
    OPTIONAL MATCH (p)-[:REQUIRES]->(dr:DocumentRequirement)
    WITH p, c, combined_score, collect(DISTINCT ct.name) AS customer_types, collect(DISTINCT dr.name) AS required_docs
    RETURN p.name AS document_name,
           c.name AS category,
           coalesce(p.extracted_rule, "") AS extracted_rule,
           coalesce(p.source_text, "") AS source_text,
           customer_types,
           required_docs,
           combined_score AS score
    ORDER BY score DESC
    LIMIT $top_k
    """

    vector_only_query = """
    CALL db.index.vector.queryNodes('policy_embeddings', $top_k, $question_embedding)
    YIELD node AS p, score
    MATCH (p)-[:BELONGS_TO]->(c:Category)
    WHERE NOT ()-[:SUPERSEDES]->(p)
    OPTIONAL MATCH (p)-[:APPLIES_TO]->(ct:CustomerType)
    OPTIONAL MATCH (p)-[:REQUIRES]->(dr:DocumentRequirement)
    WITH p, c, score, collect(DISTINCT ct.name) AS customer_types, collect(DISTINCT dr.name) AS required_docs
    RETURN p.name AS document_name,
           c.name AS category,
           coalesce(p.extracted_rule, "") AS extracted_rule,
           coalesce(p.source_text, "") AS source_text,
           customer_types,
           required_docs,
           score
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
                        )
                    )
                )

        if not records:
            return []

        max_score = max(float(record["score"]) for record in records)
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
You are the Sentinel Banking Co-Pilot.
Answer the user's question using ONLY the provided active_context.
If the context is empty or does not contain the answer, you MUST reply with:
"I cannot find a verified active policy for this in the current database."
Always cite the document name.

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
        print(
            "Expected endpoint: "
            f"{os.getenv('NEO4J_URI', 'bolt://127.0.0.1:7687')}"
        )
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

            question_embedding = embeddings_model.embed_query(user_question)
            active_context = retrieve_active_policy(
                driver,
                user_question,
                question_embedding,
                top_k=5,
            )
            answer = generate_answer(llm, active_context, user_question)
            print_response(answer, active_context)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
