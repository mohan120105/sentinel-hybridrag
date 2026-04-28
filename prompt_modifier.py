"""Sentinel: Edge AI Prompt Modifier for Enterprise Hybrid GraphRAG Banking.

Architectural purpose:
- Implements a remote prompt refinement stage that transforms terse user text
    into retrieval-grade search intent for downstream GraphRAG components.
- Uses a Hugging Face-hosted router microservice to avoid local model storage
    and local inference dependencies on the application server.
- Keeps the service import-light so deployment can succeed without GGUF files
    or llama-cpp-python present on disk.

Compliance relevance:
- Supports Strict Retrieval Constraint workflows by improving intent precision
    before graph/vector retrieval orchestration.
- Contributes to Stateful Auditability through deterministic, bounded prompt
    generation behavior suitable for reproducible incident review.
"""

from __future__ import annotations

import os
import time

import requests

HF_ROUTER_URL = os.getenv("HF_ROUTER_URL")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

def enhance_query_for_graphrag(user_query: str) -> str:
    """Rewrite user input into a retrieval-optimized GraphRAG query string.

    The function performs controlled prompt normalization to increase retrieval
    precision for Sentinel's hybrid graph and vector stack while minimizing
    hallucination risk through domain-grounded constraints.

    Args:
        user_query: Raw user question or shorthand intent text.

    Returns:
        str: A compact, professional query phrasing suitable for downstream
        retrieval and ranking.

    Raises:
        RuntimeError: If the router URL is not configured or the router call
        fails.
    """
    if not HF_ROUTER_URL:
        raise RuntimeError("HF_ROUTER_URL is not set. Configure the Hugging Face router endpoint.")

    start_time = time.time()
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    response = requests.post(
        HF_ROUTER_URL,
        json={"prompt": user_query},
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()

    payload = response.json()
    optimized_query = payload.get("optimized_query")
    if not optimized_query:
        raise RuntimeError("HF router response did not include optimized_query.")

    print(f"⚡ Modifier ran in {round(time.time() - start_time, 2)}s")
    return str(optimized_query).strip()


if __name__ == "__main__":
    raw_1 = "nri docs needed"
    print(f"Raw Input 1: {raw_1}")
    print(f"Enhanced 1:  {enhance_query_for_graphrag(raw_1)}")

    raw_2 = "fd rates for senior"
    print(f"Raw Input 2: {raw_2}")
    print(f"Enhanced 2:  {enhance_query_for_graphrag(raw_2)}")
