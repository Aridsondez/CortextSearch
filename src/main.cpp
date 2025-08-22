#include "FileScanner.hpp"
#include "ContextExtractor.hpp"
#include "EmbeddingEngine.hpp"
#include "DatabaseManager.hpp"
#include "SearchEngine.hpp"

#include <iostream>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <string>
#include <vector>

// Forward decls
void indexFiles(const std::string& path, DatabaseManager& dbManager, ContextExtractor& extractor, EmbeddingEngine& embedder);
void searchFiles(const std::string& query, DatabaseManager& dbManager, EmbeddingEngine& embedder);
std::time_t getLastModified(std::string& filePath);
bool isCorrectFileType(const std::string& extension);

// Usage helper
static void printUsage(const char* argv0) {
    std::cout << "Usage:\n"
              << "  " << argv0 << " --index  <directory_path>\n"
              << "  " << argv0 << " --search \"<query>\"\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    // --- NEW: paths for the embedding stack ---
    // These are relative to the working dir when you run the binary.
    // Use absolute paths if you prefer.
    const std::string onnxModelPath   = "models/model.onnx";
    const std::string pythonExe       = "./.venv/bin/python";
    const std::string tokenizerScript = "tools/tokenize.py";
    const std::string tokenizerJson   = "models/tokenizer.json";
    const size_t      maxSeqLen       = 256;

    // Classes
    ContextExtractor extractor;

    // --- CHANGED: EmbeddingEngine now needs model + python + tokenizer paths ---
    EmbeddingEngine embedding(
        onnxModelPath,
        pythonExe,
        tokenizerScript,
        tokenizerJson,
        maxSeqLen
    );

    DatabaseManager manager("cortex.db"); 

    // CLI
    const std::string mode  = argv[1];
    const std::string input = argv[2];

    if (mode == "--index") {
        indexFiles(input, manager, extractor, embedding);
    } else if (mode == "--search") {
        searchFiles(input, manager, embedding);
    } else {
        std::cout << "Unknown mode: " << mode << "\n";
        printUsage(argv[0]);
        return 1;
    }

    return 0;
}

void indexFiles(const std::string& path, DatabaseManager& dbManager, ContextExtractor& extractor, EmbeddingEngine& embedder) {
    FileScanner scanner;
    std::vector<FileInfo> files = scanner.scanDirectory(path);

    int indexCount = 0;
    for (const auto& file : files) {
        if (!isCorrectFileType(file.extension)) continue;

        std::string filePath = file.path;
        std::time_t lastModified = getLastModified(filePath);

        std::string context = extractor.extractText(file.path);
        if (context.empty()) {
            std::cout << "No text extracted from: " << file.name << std::endl;
            continue;
        }

        std::vector<float> embeddingVector = embedder.createEmbedding(context);
        if (embeddingVector.empty()) {
            std::cout << "Embedding failed for: " << file.name << std::endl;
            continue;
        }

        // NOTE: this call must match your DatabaseManager signature.
        // If your insertFile has different params, adjust accordingly.
        if (dbManager.insertFile(file.path, file.name, file.extension, embeddingVector, lastModified)) {
            std::cout << "Inserted/Updated " << file.path << std::endl;
            ++indexCount;
        }
    }

    std::cout << "Indexing Completed. Indexed " << indexCount << " new files." << std::endl;
}

void searchFiles(const std::string& query, DatabaseManager& dbManager, EmbeddingEngine& embedder) {
    SearchEngine searcher(dbManager, embedder);

    std::vector<SearchResult> results = searcher.search(query);
    if (results.empty()) {
        std::cout << "No Matching File Found." << std::endl;
        return;
    }

    std::cout << "\nTop matches:\n";
    for (const auto& result : results) {
        std::cout << "File: " << result.name << " (Score: " << result.score << ")\n";
        std::cout << "Path: " << result.path << "\n";
        std::cout << "--------------------------------------\n";
    }
}

bool isCorrectFileType(const std::string& extension) {
    return (extension == ".txt" || extension == ".pdf" || extension == ".png" ||
            extension == ".jpg" || extension == ".jpeg");
}

std::time_t getLastModified(std::string& filePath) {
    auto ftime = std::filesystem::last_write_time(filePath);
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ftime - std::filesystem::file_time_type::clock::now()
        + std::chrono::system_clock::now()
    );
    return std::chrono::system_clock::to_time_t(sctp);
}
