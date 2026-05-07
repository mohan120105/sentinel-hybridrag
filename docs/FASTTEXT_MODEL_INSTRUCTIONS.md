FastText language model (lid.176.bin) installation instructions

1) Recommended download (official):
   https://dl.fbaipublicfiles.com/fasttext/vectors_langid/lid.176.bin

2) Place the file in one of these locations (checked automatically by the app):
   - ./models/lid.176.bin
   - ./lid.176.bin
   - path specified by env var FASTTEXT_LANG_MODEL

3) Python packages:
   - Optional (preferred): `pip install fasttext`
     Note: `fasttext` may require a C/C++ build toolchain on Windows. If you run
     into build errors, either install a prebuilt wheel for your Python version
     or use the fallback below.

   - Fallback: `pip install langdetect`

4) If you cannot install `fasttext` on Windows, the system will automatically
   fall back to `langdetect`. `langdetect` is pure-Python but less accurate for
   some languages; it prevents crashes when `fasttext` is unavailable.

5) Example quick start:

```bash
pip install fasttext langdetect
curl -LO https://dl.fbaipublicfiles.com/fasttext/vectors_langid/lid.176.bin
mkdir -p models && mv lid.176.bin models/
```

6) Troubleshooting:
- If `pip install fasttext` fails on Windows, try to find a compatible wheel
  for your Python version or use a Linux/macOS environment. The fallback
  `langdetect` ensures the service remains functional.
