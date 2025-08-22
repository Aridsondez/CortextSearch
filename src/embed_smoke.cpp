// src/embed_smoke.cpp
#include "EmbeddingEngine.hpp"
#include <iostream>

int main() {
  // Adjust these paths if you run the binary from a different working dir.
  const char* modelPath   = "models/model.onnx";
  const char* pythonExe   = "./.venv/bin/python";   // venv python (so 'tokenizers' is available)
  const char* tokScript   = "tools/tokenize.py";    // tokenizer helper script
  const char* tokJson     = "models/tokenizer.json";
  const size_t maxSeqLen  = 256;

  try {
    EmbeddingEngine EE{
      modelPath,
      pythonExe,
      tokScript,
      tokJson,
      maxSeqLen
    };

    const std::string text = "hello world";
    auto v = EE.createEmbedding(text);
    if (v.empty()) {
      std::cerr << "Embedding failed\n";
      return 1;
    }

    std::cout << "Text: \"" << text << "\"\n";
    std::cout << "Embedding dim = " << v.size() << "\n";
    std::cout << "First 8 values:";
    for (int i = 0; i < 8 && i < static_cast<int>(v.size()); ++i) {
      std::cout << " " << v[i];
    }
    std::cout << "\n";
    return 0;

  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 2;
  }
}
