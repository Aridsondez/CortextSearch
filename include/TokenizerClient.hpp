#pragma once
#include <string>
#include <vector>
#include <optional>

// this is actually going to just call the python script in tools/tokenizer.py 
struct TokenizerResults {
    std::vector<int64_t> input_ids; // vector to hold the token IDs
    std::vector<int64_t> attention_mask; // vector to hold the attention mask
};

class TokenizerClient{

    public:
        TokenizerClient(const std::string pythonExe, std::string modelPath, std::string tokenizerJson, int maxLen = 256);
        std::optional<TokenizerResults> encode(const std::string& text) const; 


    private:
       std::string py_;
        std::string script_;
        std::string tokjson_;
        int maxLen_;
};