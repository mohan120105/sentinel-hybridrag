"""Streamlit app for Sentinel modules including Co-Pilot retrieval chat."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Sequence, Set

import streamlit as st
from init_graph import GraphAction
from neo4j import Driver
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import ValidationError

from query_copilot import (
    ActivePolicy,
    build_embeddings_model,
    build_groq_llm,
    build_neo4j_driver,
    generate_answer,
    load_environment,
    retrieve_active_policy,
)

QUESTION_STOPWORDS: Set[str] = {
    "what",
    "which",
    "when",
    "where",
    "who",
    "why",
    "how",
    "the",
    "for",
    "and",
    "are",
    "with",
    "from",
    "this",
    "that",
    "please",
}


@st.cache_resource(show_spinner=False)
def get_cached_driver() -> Driver:
    """Create and cache Neo4j driver for low-latency Streamlit interactions."""

    load_environment()
    return build_neo4j_driver()


@st.cache_resource(show_spinner=False)
def get_cached_llm() -> Any:
    """Create and cache Groq LLM client for response synthesis."""

    load_environment()
    return build_groq_llm()


@st.cache_resource(show_spinner=False)
def get_cached_embeddings_model() -> Any:
    """Create and cache embeddings model for semantic retrieval."""

    load_environment()
    return build_embeddings_model()


def ingest_graph_action_to_neo4j(
    action: GraphAction,
    document_name: str,
    issue_date: str,
    source_text: str,
) -> None:
    """Insert validated GraphAction into Neo4j with embeddings and supersession links."""

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

    driver = get_cached_driver()
    embeddings_model = get_cached_embeddings_model()
    semantic_text = f"{action.extracted_rule}\n\n{source_text}"
    embedding = [float(value) for value in embeddings_model.embed_query(semantic_text)]
    timestamp = datetime.utcnow().isoformat()

    with driver.session() as session:
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


def _normalize_graph_action_payload(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map common LLM key variants to strict GraphAction schema keys."""

    normalized: Dict[str, Any] = dict(raw_payload)

    # Key alias mapping from common extraction outputs.
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
            normalized[list_key] = [str(list_value).strip()] if str(list_value).strip() else []

    return normalized


def render_dashboard() -> None:
    """Render dashboard placeholder section."""

    st.subheader("Dashboard")
    st.info("Dashboard widgets can be added here.")


def render_curator_agent() -> None:
    """Render curator agent placeholder section."""

    st.subheader("Curator Agent")
    st.info("Curator ingestion controls can be added here.")


def render_universal_ingestion() -> None:
    """Render multimodal ingestion and directly write validated policy actions to Neo4j."""

    st.subheader("Universal Ingestion")
    st.caption(
        "Upload PDFs/images, extract governed GraphAction JSON, and ingest into graph + vectors."
    )

    load_environment()
    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    uploaded_file = st.file_uploader(
        "Upload File (PDF/Image)",
        type=["pdf", "jpg", "jpeg", "png"],
    )

    if not uploaded_file:
        return

    if not gemini_api_key:
        st.warning("GEMINI_API_KEY is missing in .env. Add it and retry.")
        return

    if st.button("Extract + Ingest", type="primary"):
        with st.spinner("Analyzing document and ingesting into Neo4j..."):
            try:
                try:
                    from google import genai
                    from google.genai import types as genai_types
                except ImportError as exc:
                    raise RuntimeError(
                        "google-genai is not installed in this deployment. "
                        "Add google-genai to requirements.txt and redeploy the app."
                    ) from exc

                prompt = (
                    # "Analyze this banking document. Extract the core policy rule. "
                    # "Determine which fixed category it belongs to (Retail_Loans, Corporate_Banking, "
                    # "KYC_AML, Credit_Cards, Tax_Compliance). Determine if it is a CREATE_NEW policy "
                    # "or if it SUPERSEDE_OLD policies. "
                    # "Identify any specific customer types this policy applies to (e.g., NRI, MSME) and "
                    # "list them in 'applies_to_customer'. Identify any specific documents or collateral "
                    # "required by this policy and list them in 'requires_document'. If none are mentioned, "
                    # "return empty lists. Return strictly in JSON format matching the schema.\n\n"
                    "Analyze this banking document. Extract the core policy rule. "
                    "CRITICAL: If the document contains a table, matrix, or numerical slabs, YOU MUST include the entire table formatted as Markdown inside the 'extracted_rule' string. Do not summarize away the numbers! "
                    "Determine which fixed category it belongs to (Retail_Loans, Corporate_Banking, KYC_AML, Credit_Cards, Tax_Compliance). "
                    "Determine if it is a CREATE_NEW policy or if it SUPERSEDE_OLD policies. "
                    "Return strictly in JSON format matching the schema.\n\n"
                    "Use ONLY this JSON schema with exact keys:\n"
                    "{\n"
                    "  \"target_node\": \"Retail_Loans|Corporate_Banking|KYC_AML|Credit_Cards|Tax_Compliance\",\n"
                    "  \"action_type\": \"CREATE_NEW|SUPERSEDE_OLD\",\n"
                    "  \"extracted_rule\": \"string\",\n"
                    "  \"superseded_document\": \"string or null\",\n"
                    "  \"applies_to_customer\": [\"string\"],\n"
                    "  \"requires_document\": [\"string\"]\n"
                    "}\n"
                    "Do not include markdown fences. Do not include extra keys."
                )

                client = genai.Client(api_key=gemini_api_key)
                file_bytes = uploaded_file.read()
                response = client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=[
                        genai_types.Content(
                            role="user",
                            parts=[
                                genai_types.Part.from_bytes(
                                    data=file_bytes,
                                    mime_type=uploaded_file.type or "application/octet-stream",
                                ),
                                genai_types.Part.from_text(text=prompt),
                            ],
                        )
                    ],
                    config=genai_types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )

                response_text = (response.text or "").strip()
                if response_text.startswith("```"):
                    response_text = response_text.strip("`")
                    response_text = response_text.replace("json", "", 1).strip()

                try:
                    payload = json.loads(response_text)
                except json.JSONDecodeError as error:
                    st.error(f"Gemini returned invalid JSON: {error}")
                    st.code(response_text)
                    return

                normalized_payload = _normalize_graph_action_payload(payload)
                action = GraphAction.model_validate(normalized_payload)

                st.success(f"Extracted rule: {action.extracted_rule}")
                st.success(f"Action type: {action.action_type}")
                st.json(action.model_dump())

                inferred_source_text = (
                    f"Multimodal ingestion from {uploaded_file.name}. "
                    f"Rule: {action.extracted_rule}"
                )
                ingest_graph_action_to_neo4j(
                    action=action,
                    document_name=uploaded_file.name,
                    issue_date=datetime.utcnow().date().isoformat(),
                    source_text=inferred_source_text,
                )

                st.success(
                    "✅ Document successfully ingested into Neo4j Knowledge Graph and Vector Index."
                )
            except ValidationError as error:
                st.error(f"Schema validation failed for GraphAction: {error}")
            except ServiceUnavailable as error:
                st.error(f"Neo4j is unavailable during ingestion: {error}")
            except Neo4jError as error:
                st.error(f"Neo4j ingestion query failed: {error}")
            except Exception as error:
                st.error(f"Universal ingestion failed: {error}")


def _format_evidence(active_context: Sequence[ActivePolicy]) -> List[str]:
    """Format citation lines for evidence snapshot display."""

    lines: List[str] = []
    for item in active_context:
        line = f"{item.document_name} [{item.category}]"

        if item.customer_types:
            line += f" | Targets: {item.customer_types}"
        if item.required_docs:
            line += f" | Requires: {item.required_docs}"

        lines.append(line)

    return lines


def _question_terms(question: str) -> Set[str]:
    """Extract lightweight lexical terms from user question for relevance filtering."""

    tokens = re.findall(r"[a-zA-Z0-9_]+", question.lower())
    return {
        token
        for token in tokens
        if len(token) > 2 and token not in QUESTION_STOPWORDS
    }


def _filter_relevant_context(
    user_question: str,
    retrieved_context: Sequence[ActivePolicy],
) -> List[ActivePolicy]:
    """Keep only retrieved policies that lexically match the user intent.

    Falls back to top-1 candidate when lexical filtering yields no match.
    """

    if not retrieved_context:
        return []

    terms = _question_terms(user_question)
    if not terms:
        return [retrieved_context[0]]

    min_overlap = 1 if len(terms) == 1 else 2
    scored: List[tuple[int, ActivePolicy]] = []

    for policy in retrieved_context:
        haystack = " ".join(
            [policy.document_name, policy.category, policy.extracted_rule, policy.source_text]
        ).lower()
        overlap_count = sum(1 for term in terms if term in haystack)
        if overlap_count >= min_overlap:
            scored.append((overlap_count, policy))

    if not scored:
        return [retrieved_context[0]]

    best_overlap = max(score for score, _ in scored)
    return [policy for score, policy in scored if score == best_overlap]


def render_copilot_retrieval() -> None:
    """Render Co-Pilot retrieval chat with strict active-policy grounding."""

    st.subheader("Co-Pilot (Retrieval)")
    st.caption(
        "Answers are grounded only in active policies. Superseded policies are excluded."
    )

    if st.button("Clear Co-Pilot Chat"):
        st.session_state.copilot_messages = []
        st.rerun()

    if "copilot_messages" not in st.session_state:
        st.session_state.copilot_messages = []

    messages: List[Dict[str, Any]] = st.session_state.copilot_messages

    for msg in messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
            evidence = msg.get("evidence", [])
            if role == "assistant" and evidence:
                with st.expander("Source Citation / Evidence Snapshot"):
                    for line in evidence:
                        st.markdown(f"- {line}")

    user_question = st.chat_input("Ask a policy question...")
    if not user_question:
        return

    messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving active policy context..."):
            try:
                driver = get_cached_driver()
                llm = get_cached_llm()
                embeddings_model = get_cached_embeddings_model()

                question_embedding = embeddings_model.embed_query(user_question)
                active_context = retrieve_active_policy(
                    driver,
                    user_question,
                    question_embedding,
                    top_k=5,
                )
                # active_context = _filter_relevant_context(user_question, retrieved_context)
                answer = generate_answer(llm, active_context, user_question)
                evidence = _format_evidence(active_context)

                st.markdown(answer)
                if evidence:
                    with st.expander("Source Citation / Evidence Snapshot"):
                        st.caption(f"{len(evidence)} active source(s)")
                        for line in evidence:
                            st.markdown(f"- {line}")
                else:
                    st.caption(
                        "Source Citation / Evidence Snapshot: "
                        "No matching active policy found."
                    )

                messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "evidence": evidence,
                    }
                )
            except ServiceUnavailable as error:
                error_message = (
                    "Neo4j is not reachable. Start Neo4j and verify Bolt connection. "
                    f"Details: {error}"
                )
                st.error(error_message)
                messages.append({"role": "assistant", "content": error_message})
            except Neo4jError as error:
                error_message = f"Neo4j query failed: {error}"
                st.error(error_message)
                messages.append({"role": "assistant", "content": error_message})
            except Exception as error:
                error_message = f"Unable to process Co-Pilot request: {error}"
                st.error(error_message)
                messages.append({"role": "assistant", "content": error_message})


def main() -> None:
    """Main Streamlit routing for Sentinel sections."""

    st.set_page_config(page_title="Sentinel", page_icon="S", layout="wide")
    st.title("Sentinel - Enterprise Agentic Data Governance")

    with st.sidebar:
        st.header("Navigation")
        selected = st.radio(
            "Select Module",
            (
                "Dashboard",
                "Curator Agent",
                "Universal Ingestion",
                "💬 Co-Pilot (Retrieval)",
            ),
        )

    if selected == "Dashboard":
        render_dashboard()
    elif selected == "Curator Agent":
        render_curator_agent()
    elif selected == "Universal Ingestion":
        render_universal_ingestion()
    else:
        render_copilot_retrieval()


if __name__ == "__main__":
    main()
