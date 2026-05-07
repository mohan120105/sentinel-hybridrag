from dotenv import load_dotenv, find_dotenv
import os
import requests
import json

# Load .env from repo root
load_dotenv(find_dotenv())

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBEDDING_ENDPOINT = os.getenv("HF_EMBEDDING_ENDPOINT")

if not HF_TOKEN:
    print("HF_TOKEN not found in environment/.env")
    raise SystemExit(2)

headers = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

metadata_url = f"https://api-inference.huggingface.co/models/{MODEL}"
emb_url = f"https://api-inference.huggingface.co/embeddings/{MODEL}"

print("Fetching model metadata:", metadata_url)
try:
    r = requests.get(metadata_url, headers=headers, timeout=20)
    print("Metadata status:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2)[:4000])
    except Exception:
        print(r.text[:4000])
except Exception as e:
    print("Metadata request failed:", e)

print("\nCalling embeddings endpoint:", emb_url)
try:
    payload = {"inputs": "probe"}
    r = requests.post(emb_url, headers=headers, json=payload, timeout=30)
    print("Embeddings status:", r.status_code)
    try:
        print(json.dumps(r.json(), indent=2)[:4000])
    except Exception:
        print(r.text[:4000])
except Exception as e:
    print("Embeddings request failed:", e)

if EMBEDDING_ENDPOINT:
    print("\nCalling deployed endpoint:", EMBEDDING_ENDPOINT)
    try:
        payload = {"inputs": "probe"}
        r = requests.post(EMBEDDING_ENDPOINT, headers=headers, json=payload, timeout=30)
        print("Deployed endpoint status:", r.status_code)
        try:
            body = r.json()
            print(json.dumps(body, indent=2)[:4000])
            if isinstance(body, list) and body:
                first = body[0]
                if isinstance(first, dict) and "embedding" in first and isinstance(first["embedding"], list):
                    print("Detected embedding dimension:", len(first["embedding"]))
                elif isinstance(first, list):
                    print("Detected embedding dimension:", len(first))
        except Exception:
            print(r.text[:4000])
    except Exception as e:
        print("Deployed endpoint request failed:", e)
else:
    print("\nHF_EMBEDDING_ENDPOINT is not set in .env; skipping deployed endpoint probe.")
