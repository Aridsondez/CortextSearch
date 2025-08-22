/*Now generate an embedding based on the actual inputs of the file contents*/

#include "EmbeddingEngine.hpp"
#include <iostream>
#include <vector>


//opening up the downloaded ai to access the model and create embeddings using such model
//currently maybe not working because i can't access the actuall library that allows me to access the model

EmbeddingEngine::EmbeddingEngine(const std::string& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "CortexSearch"),
      session(nullptr),
      sessionOptions()
{
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
        session = Ort::Session(env, modelPath.c_str(), sessionOptions);
        std::cout << "[ONNX] Model loaded successfully from: " << modelPath << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load ONNX model: " << e.what() << std::endl;
    }
}

std::vector<float> EmbeddingEngine::createEmbedding(const std::string& context){
    std::vector<float> vector;

    //alright creating the token for the input 
    std::vector<int64_t> inputIds = 

    for(int i = 0; i < 10; i++){
        vector.push_back(static_cast<float>( context.length() % (i+5)/(i+1)));
    }

    return vector;
}
