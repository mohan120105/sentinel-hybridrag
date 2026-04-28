from __future__ import annotations

import os
import time

import requests


HF_ROUTER_URL = os.getenv("HF_ROUTER_URL")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")


def enhance_query_for_graphrag(user_query: str) -> str:
    """Send the query to the remote Hugging Face router and return the optimized text."""

    if not HF_ROUTER_URL:
        raise RuntimeError("HF_ROUTER_URL is not set.")

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
    raw_input = "nri docs needed"
    print(f"Raw Input: {raw_input}")
    print(f"Enhanced:  {enhance_query_for_graphrag(raw_input)}")