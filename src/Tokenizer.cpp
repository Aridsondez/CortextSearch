#include "Tokenizer.hpp"
#include <fstream>// to read the vocabulary file
#include <sstream>// to parse the vocabulary file


Tockenizer::Tockenizer(const std::string& modelPath) {
    loadVocabulary(modelPath);
}

void Tockenizer::loadVocabulary(const std::string& modelPath) {
    std::ifstream vocabFile(modelPath);
    // Check if the file opened successfully
    if (!vocabFile.is_open()) {
        throw std::runtime_error("Failed to open vocabulary file: " + modelPath);
    }
    std::string line;
    int64_t index = 0;
    while (std::getline(vocabFile, line)) {
        // Assuming each line contains a token
        vocab[line] = index++;// create hashmapping of the token to the index
    }
}

std::vector<std::string> Tockenizer::parseToken(const std::string& text) {
    std::vector<std::string> tokens;// create a vector to store the tokens

    std::istringstream iss(text);// use istringstream to split the text into words based on whitespace

    std::string token; // to hold each token

    while (iss >> token) {
        std::transform(token.begin(), token.end(), token.begin(), ::tolower); // convert to lowercase
        tokens.push_back(token);
    }

    return tokens;
}

std::vector<int64_t> Tockenizer::tokenize(const std:: string& text) {
    std::vector<std::string> tokens = parseToken(text);
    std::vector<int64_t> tokenIds; // vector to hold the token IDs

    for (const auto& token : tokens) {
        if (vocab.find(token) != vocab.end()) {
            tokenIds.push_back(vocab[token]); // get the token ID from the vocabulary
        } else {
            tokenIds.push_back(vocab["[UNK]"]); // if token not found, use the unknown token ID
        }
    }

    return tokenIds;
}
