/*Now creating a random embedding Engine that I think takes in the input strings and returns 
A random number based on the amount of characters in the context 
Now because these are vectors it has to have multiple numbers to choose from in the vector */

#pragma once

#include <string>
#include <vector>
#include <cstdlib>
#include <onnxruntime/onnxruntime_cxx_api.h>

class EmbeddingEngine{
    public:
        EmbeddingEngine(const std::string& modelPath);
        std::vector<float> createEmbedding(const std::string& context);
    
    private:
        Ort::Env env;
        Ort::Session session;
        Ort::SessionOptions sessionOptions;

        std::vector<int64_t> tokenize(const std::string& text);

};