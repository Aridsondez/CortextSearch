### CortextSearch: AI Powered Local File Finder for macOS

# CortexSearch

CortexSearch is a C++-based local file indexing and search engine that uses contextual embeddings to allow for semantic search across PDFs, images (via OCR), and text files.

---

## Project Purpose

Build a blazing-fast, local, intelligent file search engine. CortexSearch does the following:
- Indexes text from `.txt`, `.pdf`, `.jpg`, `.png`, and `.jpeg` files.
- Extracts context using OCR (for images) and PDF parsers.
- Computes contextual embeddings from content (using ONNX model).
- Stores data and vectors in a SQLite database.
- Allows real-time semantic search via cosine similarity.

---

## Features Implemented

###  Core Components:
- **FileScanner**: Recursively scans directories, ignores duplicates (by path + last modified timestamp).
- **ContextExtractor**: Extracts text from files using Tesseract for images and Poppler for PDFs.
- **EmbeddingEngine (Stub)**: Currently generates dummy/random embeddings for extracted text.
- **DatabaseManager**: Uses SQLite3 to store file info and vector embeddings. Supports insert/update logic.
- **SearchEngine**: Computes cosine similarity and ranks search results by match strength.

### Duplication Detection:
- Uses full **absolute paths** and **last modified timestamps** to skip or update changed files.

### CLI Application:
```bash
# Index a directory
./CortexSearch --index /path/to/files

# Search with a query
./CortexSearch --search "project plan for solar"
```

---

## What’s Left To Do

### 1. Real Embedding (In Progress)
- Install ONNX Runtime via Homebrew
- Implement tokenizer to convert input text → model input tensor
- Feed tokenized input into ONNX model to get real embedding vector

### 2. Tokenizer
- Must match the ONNX model's expectations (e.g., BERT tokenizer)
- Outputs `input_ids`, `attention_mask` for ONNX input

### 3. C++ UI (Optional)
- Show indexed files, embeddings, and search results in a graphical interface
- Can be done with **Qt** (full-featured) or **Dear ImGui** (lightweight dev dashboard)

---

## 🛠️ Tech Stack

| Area | Tool/Lib |
|------|----------|
| Language | C++17 |
| DB | SQLite3 |
| Embedding | ONNX Runtime (Transformer model) |
| PDF Parsing | Poppler |
| OCR | Tesseract |
| Search | Cosine Similarity |
| Build System | CMake |
| CLI Interface | Custom Main Driver |
| GUI (planned) | Qt or Dear ImGui |

---

## Concepts To Remember

- **Tokenizer** prepares raw text for the model (splits into tokens, adds padding, etc.).
- **ONNX Model** outputs dense vector embeddings from tokenized input.
- **Search** works by comparing query vector to stored file vectors using cosine similarity.
- All file paths are normalized to absolute paths to avoid duplicates.

---

## Project Structure

```
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
│   └── model.onnx
├── testData/
│   └── example files for testing
├── build/
├── CMakeLists.txt
└── README.md  <-- (this file)
```

---

## Commands To Remember

```bash
# Build the project
cmake -S . -B build
cmake --build build

# Run the program
./build/CortexSearch --index /path/to/files
./build/CortexSearch --search "meeting notes from june"

# Check ONNX path manually
ls /opt/homebrew/opt/onnxruntime/include/onnxruntime
```

---

## Next Steps

1. Finish `Tokenizer` (text → tensor inputs)
2. Complete `createEmbedding()` with ONNX model inference
3. Add basic C++ GUI (if needed) to view file database + query results
4. Refactor if needed for deployment (static linking, resource bundling)
