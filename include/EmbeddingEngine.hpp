/*Now creating a random embedding Engine that I think takes in the input strings and returns 
A random number based on the amount of characters in the context 
Now because these are vectors it has to have multiple numbers to choose from in the vector */

#pragma once

#include <string>
#include <vector>
#include <cstdlib>
#include "TokenizerClient.hpp"
#include <memory>
#include <onnxruntime/onnxruntime_cxx_api.h>

class EmbeddingEngine{
    public:
       EmbeddingEngine(const std::string& onnxModelPath,
                    const std::string& pythonExe,          // ./.venv/bin/python
                    const std::string& tokenizerScript,    // tools/tokenize.py
                    const std::string& tokenizerJson,      // models/tokenizer.json
                    size_t maxSeqLen = 256);

        std::vector<float> createEmbedding(const std::string& text);

    private:
        Ort::Env env;
        Ort::Session session{nullptr};
        Ort::SessionOptions sessionOptions;

        std::vector<std::string> inputNamesOwned_;
        std::vector<std::string> outputNamesOwned_;

        std::unique_ptr<TokenizerClient> tok_;

        size_t maxSeqLen_;

};