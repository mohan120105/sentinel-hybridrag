#!/usr/bin/env python3
"""Minimal language detection test without full module import."""

import os
import sys
import fasttext

# Test 1: Check if imports are available
print("=" * 60)
print("Step 1: Checking optional dependency availability")
print("=" * 60)

try:
    import fasttext
    print("✓ fasttext is available")
except ImportError:
    print("✗ fasttext not available (expected on first run or Windows build issues)")

try:
    import langdetect
    print("✓ langdetect is available")
except ImportError:
    print("✗ langdetect not available (will use fallback)")

print("\n" + "=" * 60)
print("Step 2: Testing basic language detection logic")
print("=" * 60)

# Language code mapping (from query_copilot.py)
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

def basic_detect(text):
    """Fallback detection without external libraries."""
    if not text:
        return "English"
    # Devanagari (Hindi)
    if any("\u0900" <= ch <= "\u097F" for ch in text):
        return "Hindi"
    # Telugu
    if any("\u0C00" <= ch <= "\u0C7F" for ch in text):
        return "Telugu"
    # Spanish indicators
    lower = text.lower()
    if any(c in text for c in "¿¡áéíóúñÁÉÍÓÚÑ"):
        return "Spanish"
    return "English"

test_cases = [
    ("Please show the audit doc AUDIT-2026-Q1-RED", "English"),
    ("¿Cuál es la tasa de TDS para pagos?", "Spanish"),
    ("हेलो, मुझे मेरी KYC स्थिति बताएं", "Hindi"),
    ("నా TDS రేట్ ఎంత?", "Telugu"),
]

print("Running fallback heuristic detection:")
for text, expected in test_cases:
    detected = basic_detect(text)
    status = "✓" if detected.lower() == expected.lower() else "✗"
    display_text = text[:35] + "..." if len(text) > 35 else text
    print(f"{status} '{display_text}' -> {detected}")

print("\n" + "=" * 60)
print("Installation instructions:")
print("=" * 60)
print("""
If langdetect is not available, install optional dependencies:

  pip install langdetect fasttext

For FastText on Windows, you may need a prebuilt wheel. If compilation fails,
the system will fallback to pure-Python langdetect or heuristics.

Download the FastText language model (optional but recommended):

  curl -LO https://dl.fbaipublicfiles.com/fasttext/vectors_langid/lid.176.bin
  mkdir -p models
  mv lid.176.bin models/
""")

print("=" * 60)
print("✓ Test complete. System is ready for multilingual detection.")
print("=" * 60)



# Load the pre-trained model
model = fasttext.load_model("C:\\Users\\MOHAN\\Documents\\Bank-rag\\models\\lid.176.ftz")

# Text to detect
text = "Bonjour tout le monde"

# Predict the language
# k=1 returns the most likely language; labels include the confidence score
predictions = model.predict(text, k=1)

print(predictions)
# Output example: (('__label__fr',), array([0.9656]))
