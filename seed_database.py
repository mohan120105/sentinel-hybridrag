"""Baseline mass-ingestion loader for Sentinel Neo4j knowledge graph.

This script scans ./v1_baseline_docs for PDF/PNG files, extracts structured
GraphAction data using Gemini multimodal input, validates via Pydantic, and
ingests each baseline policy into Neo4j.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

from init_graph import CATEGORY_VALUES, GraphAction

try:
    from app import ingest_graph_action_to_neo4j
except Exception as import_error:  # pragma: no cover - import is environment-dependent
    raise RuntimeError(
        "Failed to import ingest_graph_action_to_neo4j from app.py. "
        "Ensure app.py is present and dependencies are installed."
    ) from import_error

BASELINE_DIR = Path("C:\\Users\\MOHAN\\Documents\\Bank-rag\\v1_baseline_docs")
SUPPORTED_EXTENSIONS = {".pdf", ".png"}
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"
RATE_LIMIT_SECONDS = 3

# --- THE BOUNCER: Absolute Source of Truth for Ontology ---
APPROVED_CATEGORIES = {
    "Retail_Loans",
    "Corporate_Banking",
    "KYC_AML",
    "Credit_Cards",
    "Tax_Compliance",
    "Foreign_Exchange_FEMA",
    "Digital_Payments_UPI",
    "Risk_Management",
    "Priority_Sector_Lending",
    "Audit_And_Inspection"
}

def _load_environment() -> None:
    """Load .env and normalize quoted environment values."""
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path=dotenv_path, override=True)

    for key in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_MULTIMODAL_MODEL"):
        value = os.getenv(key)
        if value is not None:
            os.environ[key] = value.strip().strip('"').strip("'")

def _normalize_graph_action_payload(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map common Gemini key aliases into strict GraphAction schema fields."""
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

    # Baseline ingestion is always CREATE_NEW.
    normalized["action_type"] = "CREATE_NEW"

    target_node = str(normalized.get("target_node", "")).strip()
    if target_node not in CATEGORY_VALUES:
        normalized["target_node"] = target_node

    superseded_value = normalized.get("superseded_document")
    if isinstance(superseded_value, str) and superseded_value.strip().lower() in {
        "none", "null", "n/a", "na", "",
    }:
        normalized["superseded_document"] = None

    if "superseded_document" not in normalized:
        normalized["superseded_document"] = None

    for list_key in ("applies_to_customer", "requires_document"):
        list_value = normalized.get(list_key)
        if list_value is None:
            normalized[list_key] = []
        elif isinstance(list_value, str):
            normalized[list_key] = [list_value.strip()] if list_value.strip() else []
        elif isinstance(list_value, list):
            normalized[list_key] = [str(item).strip() for item in list_value if str(item).strip()]
        else:
            normalized[list_key] = [str(list_value).strip()] if str(list_value).strip() else []

    return normalized

def _build_prompt() -> str:
    """Build strict ontology-constrained extraction prompt for baseline seeding."""
    return (
        "You are Sentinel Curator Agent for Tier-1 banking governance. "
        "Analyze this policy document and extract one actionable rule suitable for compliance retrieval. "
        "Classify into exactly one fixed ontology category. "
        "This run is baseline seeding, so action_type MUST be CREATE_NEW and superseded_document MUST be null.\n\n"
        "Identify any specific customer types this policy applies to (e.g., NRI, MSME) and "
        "list them in 'applies_to_customer'. Identify any specific documents or collateral required "
        "by this policy and list them in 'requires_document'. If none are mentioned, return empty lists.\n\n"
        "Allowed ontology categories:\n"
        "- Retail_Loans\n"
        "- Corporate_Banking\n"
        "- KYC_AML\n"
        "- Credit_Cards\n"
        "- Tax_Compliance\n"
        "- Foreign_Exchange_FEMA\n"
        "- Digital_Payments_UPI\n"
        "- Risk_Management\n"
        "- Priority_Sector_Lending\n"
        "- Audit_And_Inspection\n\n"
        "Return strict JSON only with exact keys and no markdown fences:\n"
        "{\n"
        "  \"target_node\": \"Retail_Loans|Corporate_Banking|KYC_AML|Credit_Cards|Tax_Compliance|Foreign_Exchange_FEMA|Digital_Payments_UPI|Risk_Management|Priority_Sector_Lending|Audit_And_Inspection\",\n"
        "  \"action_type\": \"CREATE_NEW\",\n"
        "  \"extracted_rule\": \"string\",\n"
        "  \"superseded_document\": null,\n"
        "  \"applies_to_customer\": [\"string\"],\n"
        "  \"requires_document\": [\"string\"]\n"
        "}\n"
        "Do not include extra keys."
    )

def _mime_type_for(file_path: Path) -> str:
    """Infer MIME type for supported file extensions."""
    extension = file_path.suffix.lower()
    if extension == ".pdf":
        return "application/pdf"
    if extension == ".png":
        return "image/png"
    return "application/octet-stream"

def _extract_graph_action_from_file(file_path: Path, model_name: str) -> GraphAction:
    """Call Gemini multimodal API and validate the result as GraphAction."""
    from google import genai
    from google.genai import types as genai_types

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is missing. Add it to your environment/.env.")

    client = genai.Client(api_key=gemini_api_key)
    file_bytes = file_path.read_bytes()

    response = client.models.generate_content(
        model=model_name,
        contents=[
            genai_types.Content(
                role="user",
                parts=[
                    genai_types.Part.from_bytes(
                        data=file_bytes,
                        mime_type=_mime_type_for(file_path),
                    ),
                    genai_types.Part.from_text(text=_build_prompt()),
                ],
            )
        ],
        config=genai_types.GenerateContentConfig(response_mime_type="application/json"),
    )

    response_text = (response.text or "").strip()
    if not response_text:
        raise ValueError("Gemini returned an empty response.")

    if response_text.startswith("```"):
        response_text = response_text.strip("`")
        response_text = response_text.replace("json", "", 1).strip()

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"Gemini returned invalid JSON: {error}") from error

    normalized_payload = _normalize_graph_action_payload(payload)
    return GraphAction.model_validate(normalized_payload)

def _collect_input_files() -> List[Path]:
    """Return sorted list of baseline document files to ingest."""
    if not BASELINE_DIR.exists() or not BASELINE_DIR.is_dir():
        raise FileNotFoundError(
            f"Baseline directory not found: {BASELINE_DIR.resolve()}\n"
            "Create ./v1_baseline_docs and add .pdf/.png files."
        )

    files = [
        path
        for path in BASELINE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    files.sort(key=lambda p: p.name.lower())
    return files

def main() -> None:
    """Seed Neo4j graph by extracting and ingesting baseline policy documents."""
    _load_environment()
    model_name = (
        os.getenv("GEMINI_MULTIMODAL_MODEL", "").strip()
        or os.getenv("GEMINI_MODEL", "").strip()
        or DEFAULT_GEMINI_MODEL
    )

    try:
        files = _collect_input_files()
    except Exception as error:
        print(f"Setup error: {error}")
        return

    if not files:
        print(f"No baseline files found in {BASELINE_DIR.resolve()} (expected .pdf or .png).")
        return

    total = len(files)
    success_count = 0

    print(f"Found {total} files in {BASELINE_DIR.resolve()}")
    print(f"Using Gemini model: {model_name}")

    for index, file_path in enumerate(files, start=1):
        print(f"Ingesting {index}/{total}: {file_path.name}...", end=" ")
        try:
            action = _extract_graph_action_from_file(file_path=file_path, model_name=model_name)

            # --- THE APPLICATION GUARDRAIL ---
            if action.target_node not in APPROVED_CATEGORIES:
                print(f"\nSECURITY ALERT: Gemini Hallucinated an invalid category '{action.target_node}'. Ingestion aborted for this file.")
                continue # Skips to the next document

            inferred_source_text = (
                f"Baseline ingestion from {file_path.name}. "
                f"Rule: {action.extracted_rule}"
            )
            ingest_graph_action_to_neo4j(
                action=action,
                document_name=file_path.name,
                issue_date=datetime.utcnow().date().isoformat(),
                source_text=inferred_source_text,
                access_code=2,
            )
            success_count += 1
            print("Success")
        except ValidationError as error:
            print(f"Failed (schema validation): {error}")
        except Exception as error:
            print(f"Failed: {error}")

        if index < total:
            time.sleep(RATE_LIMIT_SECONDS)

    print(
        f"Completed baseline seeding. Success: {success_count}/{total}, "
        f"Failed: {total - success_count}/{total}."
    )

if __name__ == "__main__":
    main()