/*creating the search engine first manager. 
-Get the input from the user 
-Create a vector embedding for the input
-Get the vector embeddings or information from each saved file
-compare the vectors to recieve similirity 
-return the most similar vectors*/

#pragma once

#include <vector>
#include <string>
#include "DatabaseManager.hpp"
#include "EmbeddingEngine.hpp"


struct SearchResult{    
    std::string path;
    std::string name;
    std::string extension;
    float score;//the closeness to the vector
};

class SearchEngine{
    public:
        //constructor takes in the databse and the Embedding vector 
        SearchEngine(DatabaseManager& manager, EmbeddingEngine& embedder);


        //search function gets the topK search results based on the input 
        std::vector<SearchResult> search(const std::string& searchInput, int topK = 5);
    
    private:
        DatabaseManager& manager;
        EmbeddingEngine& embedder;

        float cosineSimilarity(const std::vector<float>& fileEmbeddingVector, const std::vector<float> searchEmbeddingVector);

};