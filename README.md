# CortexSearch: AI-Powered Local File Finder for macOS

CortexSearch is a C++17 local semantic search engine that indexes your files, extracts context, generates embeddings with ONNX Runtime, and lets you run natural-language queries against your own data — fully offline.


https://github.com/user-attachments/assets/188a6f74-76e8-4259-8409-b8035df887f8


## Current Capabilities

Recursive File Scanning
Walks directories, avoids duplicates using absolute path + last modified timestamp.

Context Extraction

`.txt` → plain text read

`.pdf` → parsed via Poppler

`.jpg/.png/.jpeg` → OCR via Tesseract

### Real Embeddings

Integrated ONNX Runtime with a transformer model.

Implemented custom C++ tokenizer to prepare input (input_ids, attention_mask).

Generates dense embeddings per file (no more stubs).

Database Persistence (SQLite3)

Stores file metadata (path, name, extension, last_modified) + serialized embeddings.

Smart logic: INSERT new, UPDATE if modified, skip otherwise.

Search Engine

Cosine similarity between query vector and stored embeddings.

Returns ranked file matches with similarity scores.

CLI Application

# Index a directory
./CortexSearch --index /path/to/files

# Query semantically
./CortexSearch --search "project plan for solar"

🛠️ Tech Stack
Area	Tool/Lib
Language	C++17
Embeddings	ONNX Runtime (transformer model)
Tokenizer	Custom C++ BERT-style tokenizer
DB	SQLite3
PDF Parsing	Poppler
OCR	Tesseract
Search	Cosine Similarity
Build System	CMake
CLI	Custom Main Driver
GUI (optional)	Qt or Dear ImGui
## Project Structure
CortexSearch/
├── include/
│   ├── ContextExtractor.hpp
│   ├── DatabaseManager.hpp
│   ├── EmbeddingEngine.hpp
│   ├── FileScanner.hpp
│   └── SearchEngine.hpp
├── src/
│   ├── main.cpp
│   ├── ContextExtractor.cpp
│   ├── DatabaseManager.cpp
│   ├── EmbeddingEngine.cpp
│   ├── FileScanner.cpp
│   └── SearchEngine.cpp
├── models/
│   ├── model.onnx
│   └── vocab.txt
├── testData/
│   └── sample files
├── build/
├── CMakeLists.txt
└── README.md


# Build & Run
cmake -S . -B build
cmake --build build

# Index directory
./build/CortexSearch --index /Users/you/Documents

# Search
./build/CortexSearch --search "resume draft with internship"

## Next Steps

GUI (optional)
Add a simple interface for browsing indexed files & search results.
Options:

Qt → full desktop app

Dear ImGui → lightweight developer dashboard

Ranking Enhancements
Improve retrieval (hybrid keyword + semantic, configurable thresholds).

Packaging
Static build, Homebrew formula, or .dmg app bundle for macOS.

Notes

Duplicate ONNX schema warnings have been silenced (ORT_LOGGING_LEVEL_ERROR).

Works offline: all embeddings + search happen locally.

Database is persistent (cortex.db) and can be inspected via any SQLite client.

