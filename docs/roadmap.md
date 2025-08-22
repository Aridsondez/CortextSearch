# CortexSearch — Tokenizer + Embedding Integration Roadmap (Python-helper path)

This roadmap gives you a **step-by-step**, **do-this-next** plan to wire up a real tokenizer (via a tiny Python helper) and the **all-MiniLM-L6-v2** ONNX sentence embedding model. It also covers database migrations (no more deleting the DB), and tests to verify everything end-to-end.

---

## 0) Target directory layout

```
CortexSearch/
├── include/
│   ├── ContextExtractor.hpp
│   ├── DatabaseManager.hpp
│   ├── EmbeddingEngine.hpp
│   ├── FileScanner.hpp
│   ├── SearchEngine.hpp
│   └── TokenizerClient.hpp      # NEW: C++ wrapper that talks to Python helper
├── src/
│   ├── main.cpp
│   ├── ContextExtractor.cpp
│   ├── DatabaseManager.cpp
│   ├── EmbeddingEngine.cpp
│   ├── FileScanner.cpp
│   ├── SearchEngine.cpp
│   └── TokenizerClient.cpp      # NEW
├── models/
│   ├── all-MiniLM-L6-v2.onnx    # ONNX model (with pooling+normalize)
│   └── tokenizer.json           # Matching tokenizer spec
├── tools/
│   └── tokenize.py              # NEW: Python helper (HF tokenizers) -> JSON
├── tests/                       # NEW: C++ + shell tests
│   ├── test_embedding.cpp
│   ├── test_db_migration.cpp
│   ├── test_index_and_search.cpp
│   ├── test_tokenizer.sh
│   └── fixtures/
│       ├── tiny.txt
│       ├── short.pdf
│       └── image_text.png
├── build/
├── CMakeLists.txt
└── ROADMAP.md (this file)
```

---

## 1) Install prerequisites

### 1.1 ONNX Runtime (macOS / Homebrew)
```bash
brew install onnxruntime
```

### 1.2 Python environment for the tokenizer helper
Use a virtualenv so it’s isolated and reproducible:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tokenizers==0.15.*
```
> We intentionally use **`tokenizers`** (the fast Rust-backed library), **not** `transformers`. This gives you the exact subword splits + special tokens + padding that MiniLM expects.

### 1.3 Model + tokenizer files
- Put **`all-MiniLM-L6-v2.onnx`** and **`tokenizer.json`** under `./models/`.
- These two **must match** (MiniLM-L6-v2 family). The ONNX variant you’re using already includes pooling + normalization, so it outputs a ready **[1, 384]** sentence embedding.

---

## 2) Why we’re using a Python helper (plain English)

- BERT-family models (MiniLM) don’t split on whitespace; they use **subword tokenization** plus **special tokens** (`[CLS]`, `[SEP]`), **padding** to a fixed length, and an **attention mask**.
- The **tokenizers** library loads `tokenizer.json` and outputs **exact** `input_ids` + `attention_mask` using the same rules the model was trained with.
- We’ll call a small Python script from C++ to get those arrays as JSON. It’s simple, reliable, and avoids a Rust build while you get the pipeline working.

**Result:** Correct inputs → meaningful 384‑D sentence embeddings from ONNX. (Embedding size is **fixed** by the model, not by the file.)

---

## 3) Build the Python tokenizer helper

Create `tools/tokenize.py`:

```python
#!/usr/bin/env python3
import sys, json, argparse
from tokenizers import Tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-json", required=True, help="Path to tokenizer.json")
    ap.add_argument("--text", help="Text to tokenize. If omitted, read stdin.")
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer_json)

    def encode_one(text):
        enc = tok.encode(text)
        ids  = enc.ids
        mask = getattr(enc, "attention_mask", None) or [1]*len(ids)
        if len(ids) > args.max_len:
            ids = ids[:args.max_len]
            mask = mask[:args.max_len]
        elif len(ids) < args.max_len:
            pad_len = args.max_len - len(ids)
            ids  = ids + [0]*pad_len
            mask = mask + [0]*pad_len
        return {"input_ids": ids, "attention_mask": mask}

    if args.text is not None:
        payload = encode_one(args.text)
        print(json.dumps(payload))
        return

    for line in sys.stdin:
        text = line.rstrip("\n")
        payload = encode_one(text)
        print(json.dumps(payload), flush=True)

if __name__ == "__main__":
    main()
```

Make it executable:
```bash
chmod +x tools/tokenize.py
```

**Quick sanity test:**
```bash
. .venv/bin/activate
echo "hello world" | python3 tools/tokenize.py --tokenizer-json models/tokenizer.json
```

---

## 4) Add a tiny C++ wrapper: `TokenizerClient`

**Goal:** Hide the Python call behind a simple C++ API that returns `std::vector<int64_t>` for `input_ids` and `attention_mask`.

### 4.1 MVP design (per-call helper)
- Build a `std::string` command like:
  ```
  <path-to-python> tools/tokenize.py --tokenizer-json models/tokenizer.json --text "<escaped>"
  ```
- Run it with `popen` (or `std::system()` + temp file), read stdout (a single JSON line).
- Parse JSON (e.g., with `nlohmann/json.hpp`) into two `std::vector<int64_t>`.

**Contract:** Always return `input_ids.size() == attention_mask.size() == 256`.

> Keep a guard: if Python exits non‑zero or JSON parse fails, return an error to the caller (so CLI can print a human message).

### 4.2 Optional upgrade (persistent mode)
- Start the Python script once without `--text`, keep the process open.
- Write one raw text line to its stdin per request; read one JSON line per response.
- This avoids Python startup overhead for every encode.

---

## 5) Update `EmbeddingEngine` (ONNX inference)

**What changes from your stub:**
- **Before:** you returned a random vector based on string length.
- **Now:** you will call `TokenizerClient`, convert `input_ids` and `attention_mask` (both int64) into ONNX tensors of shape `[1, 256]`, run the session, and read a **384‑D** embedding.

**Implementation checklist:**
1. **Session setup** once at startup:
   - Create `Ort::Env`, `Ort::SessionOptions`, `Ort::Session` with `models/all-MiniLM-L6-v2.onnx`.
   - Print discovered **input names** and **output names** once (to log). You should see names containing `input_ids` and `attention_mask`. If a third input `token_type_ids` exists, feed zeros of shape `[1, 256]` (int64).
2. **Run per text:**
   - Call `TokenizerClient` → `ids(256)`, `mask(256)`.
   - Create ONNX tensors: shape `[1, 256]`, dtype `int64`.
   - `session.Run(...)` → you should get output `[1, 384]` (float32).
   - Copy to `std::vector<float>` (length 384). If you want, L2‑normalize again (harmless if already normalized).
3. **Return embedding** to the caller.

**Invariant:** Embedding size is **fixed (384)**. Store and validate once.

---

## 6) Database schema & migrations (stop deleting the DB)

### 6.1 Use `PRAGMA user_version` for schema versioning
- On startup, read `PRAGMA user_version`.
- If `user_version < CODE_SCHEMA_VERSION`, run migrations to bring DB up to date, then set `PRAGMA user_version = CODE_SCHEMA_VERSION`.
- Migrations must be **idempotent** and **non-destructive** (no dropping data).

### 6.2 Target schema (SQLite)

```sql
CREATE TABLE IF NOT EXISTS metadata (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  abs_path TEXT NOT NULL UNIQUE,
  last_modified INTEGER NOT NULL,
  content_hash TEXT,
  title TEXT,
  size_bytes INTEGER
);

CREATE TABLE IF NOT EXISTS embeddings (
  file_id INTEGER PRIMARY KEY,
  vector BLOB NOT NULL,
  FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);
```

**Store model info once:**
```sql
INSERT OR REPLACE INTO metadata(key, value) VALUES
  ('model_name', 'all-MiniLM-L6-v2-onnx'),
  ('embedding_dim', '384'),
  ('max_seq_len', '256');
```

### 6.3 Migration pattern examples
```sql
ALTER TABLE files ADD COLUMN size_bytes INTEGER;  -- only if it doesn't exist
PRAGMA user_version = 2;

CREATE TABLE IF NOT EXISTS file_tags (
  file_id INTEGER NOT NULL,
  tag TEXT NOT NULL,
  PRIMARY KEY(file_id, tag),
  FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
);
PRAGMA user_version = 3;
```

### 6.4 Embedding storage as BLOB
- Serialize 384 `float32` into a byte array and store in `embeddings.vector`.
- On read, ensure blob size == `384 * 4` bytes; otherwise treat as corrupt and re-embed.
- Use transactions during bulk inserts for speed.

### 6.5 Upsert logic
- When scanning a path:
  - If `abs_path` not in `files`, `INSERT` new row then `INSERT` embedding.
  - If exists but `last_modified` increased (or `content_hash` changed), re-extract text → re-embed → `UPDATE` embedding.
- Consider an index on `files(abs_path)` (unique) for lookup speed.

---

## 7) End-to-end flow glue (index & search)

**Indexing path:**
1. `FileScanner` finds files.
2. `ContextExtractor` returns text (PDF via Poppler, images via Tesseract OCR).
3. `EmbeddingEngine.createEmbedding(text)` returns 384‑D vector.
4. `DatabaseManager` upserts file row and embedding blob.

**Search path:**
1. User query → `createEmbedding(query_text)`.
2. Read all file vectors (or a candidate subset) from DB.
3. Compute cosine similarity and rank.
4. Return top‑k with scores + file metadata.

> Later improvement: **chunking** large docs (e.g., 512–1000 chars) so retrieval points to sections, not whole files.

---

## 8) Tests (to ensure each piece works)

### 8.1 Tokenizer helper tests (shell)
- **Echo test:** pipe known strings and ensure JSON parses; arrays length == 256.
- **Determinism:** same input → same `input_ids` (byte-for-byte).
- **Padding check:** very short input should end with many zeros in `input_ids` and `attention_mask`.

`tests/test_tokenizer.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
OUT=$(echo "The quick brown fox." | python3 tools/tokenize.py --tokenizer-json models/tokenizer.json)
python3 - << 'PY'
import json,sys
d=json.loads(sys.argv[1])
assert len(d["input_ids"])==256
assert len(d["attention_mask"])==256
print("Tokenizer OK:", d["input_ids"][:8], "...")
PY
"$OUT"
echo "Tokenizer helper PASS"
```

### 8.2 C++ Embedding tests
- **Shape test:** Returned embedding has length **384**.
- **Norm test:** If model normalizes, L2 norm ≈ 1 (±1e‑2). Otherwise, normalize in code and check norm == 1.
- **Determinism:** same text → same vector (within small epsilon).

Catch2 (CMake FetchContent) example:
```cmake
include(FetchContent)
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.6.0
)
FetchContent_MakeAvailable(catch2)
add_executable(tests
  tests/test_embedding.cpp
  tests/test_db_migration.cpp
  tests/test_index_and_search.cpp
)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain onnxruntime)
```

`tests/test_embedding.cpp` (skeleton):
```cpp
#include <catch2/catch_all.hpp>
#include "EmbeddingEngine.hpp"

TEST_CASE("Embedding shape & norm") {
  EmbeddingEngine ee("models/all-MiniLM-L6-v2.onnx", /* tokenizer used via TokenizerClient */);
  auto v1 = ee.createEmbedding("hello world");
  REQUIRE(v1.size() == 384);
  double s=0; for (auto f : v1) s += f*f;
  s = std::sqrt(s);
#if 1
  REQUIRE(s == Approx(1.0).margin(1e-2));
#endif
}
```

### 8.3 DB tests
- **Migration test:** Start with a temp DB at `user_version=0` → run app init → verify tables exist and `metadata` rows are set.
- **Upsert test:** Insert a file, then simulate a modified timestamp → confirm `files.last_modified` updates and `embeddings.vector` gets replaced.
- **Blob size test:** Read back vector blob and confirm size is `384 * 4` bytes.

### 8.4 Index + search integration
- Put a couple of small files with known phrases in `tests/fixtures`.
- Index them; run a search query that should hit one file strongly.
- Confirm cosine similarity ranks the expected file first.
- Add a second file with similar content; ensure both appear and ordering makes sense.

---

## 9) Build & run commands

```bash
# Build C++
cmake -S . -B build
cmake --build build

# (One-time) Activate Python venv whenever you run the helper
source .venv/bin/activate

# Quick tokenizer smoke test
echo "hello world" | python3 tools/tokenize.py --tokenizer-json models/tokenizer.json | head -c 200; echo

# Index
./build/CortexSearch --index /path/to/files

# Search
./build/CortexSearch --search "project plan for solar"
```

**Tip:** Log the model & tokenizer paths at startup, plus the discovered ONNX input/output names once.

---

## 10) Troubleshooting & gotchas

- **Tokenizer mismatch** → garbage embeddings. Always use the **matching** `tokenizer.json` for the chosen model.
- **Max length** must be consistent across index & query (256). If you change it later, re-index.
- **Embedding dim is fixed** by the model (here 384). Store and verify at startup.
- **Performance:** Per-call Python is OK to ship initially. For speed, run Python in a persistent mode (read stdin lines, write JSON lines) or later switch to the native Rust C API.
- **Error handling:** If Python helper fails or JSON is invalid, surface an actionable error (and skip the file or retry once).
- **Re-index policy:** If you change model or tokenizer, bump metadata and re-index the corpus so vectors live in the same space.

---

## 11) Minimal acceptance criteria (you know you’re done when…)

- `tools/tokenize.py` returns 256‑length `input_ids` and `attention_mask` on any input.
- `EmbeddingEngine::createEmbedding()` returns a 384‑float vector; norm ≈ 1.
- The DB has `metadata` rows (`model_name`, `embedding_dim`, `max_seq_len`), and embeddings are stored as 384‑float blobs.
- Running `--index` then `--search` returns sensible, repeatable rankings.
- Tests pass: tokenizer shape/determinism, embedding shape/norm, DB migration, and a basic end‑to‑end search case.

---

## 12) Stretch goals (optional)

- **Persistent Python helper**: start once, stream texts over stdin, get JSON per line (cut startup overhead).
- **Chunking**: long files → split into chunks (e.g., 512–1000 chars) and store multiple embeddings per file; return top‑k chunks.
- **Hybrid search**: combine cosine with keyword scoring (FTS5) for best of both worlds.
- **GUI** (Dear ImGui): show indexed files, embeddings info, and live query results.
- **Native tokenizer**: replace Python helper with Hugging Face tokenizers C API for an all‑C++ binary.
