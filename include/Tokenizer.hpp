#pragma once

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <unordered_map>//to store the tockens key value pairs

//Tockenizers basically turn text into tokens that can be used by the model
//Think what do i need for the tockenizer.


class Tockenizer{

    public:
        Tockenizer(const std::string& modelPath);
        std::vector<int64_t> tokenize(const std:: string& text);

    private:
        std::unordered_map<std::string, int64_t> vocab;
        void loadVocabulary(const std::string& modelPath);
        std::vector<std::string> parseToken(const std::string& text);
};