#!/usr/bin/env python3
"""Quick test of language detection without interactive input."""

from query_copilot import detect_user_language

# Test cases
test_cases = [
    ("Please show the audit doc AUDIT-2026-Q1-RED", "English"),
    ("¿Cuál es la tasa de TDS para pagos?", "Spanish"),
    ("हेलो, मुझे मेरी KYC स्थिति बताएं", "Hindi"),
    ("నా TDS రేట్ ఎంత?", "Telugu"),
]

print("Testing language detection:")
print("-" * 60)

for text, expected in test_cases:
    detected = detect_user_language(text)
    status = "✓" if detected.lower() == expected.lower() else "✗"
    print(f"{status} '{text[:40]}...' -> {detected} (expected: {expected})")

print("-" * 60)
print("Language detection test complete.")
