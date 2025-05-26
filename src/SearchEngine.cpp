/*Defining the search engine
--First define the constructor to get the db using sqlite3 
--Create the embedding vector for the searchInput
--Initialize the database for searching 
--go through the database getting each file
--compare the filed serialized vector to search vector
--rank the comparisons based on similiarity 
--sort and return the top searchs*/

#include "SearchEngine.hpp"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <sqlite3.h>
#include <iostream>

SearchEngine::SearchEngine(DatabaseManager& manager, EmbeddingEngine& embedder) 
    : manager(manager), embedder(embedder) {}

std::vector<SearchResult> SearchEngine::search(const std::string& searchInput, int topK){
    std::vector<SearchResult> results;
    std::vector<float> searchInputVectorEmbedding = embedder.createEmbedding(searchInput);

    //now use sqlite to go through all the files in database
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<float> >> files = manager.getAllFiles();

    //loop through all the files and deserialize the numbers
    for (const auto& [path, name, extension, serializedEmbedding] : files) {
       
        // Similarity
        float score = cosineSimilarity(searchInputVectorEmbedding, serializedEmbedding);
    
        results.push_back({path, name, extension, score});
    }

    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b)
        {
        return a.score > b.score;
        }
    );

    if (results.size() > static_cast<size_t>(topK)) {
        results.resize(topK);
    }

    return results;

}

float SearchEngine::cosineSimilarity(const std::vector<float>& fileEmbeddingVector, const std::vector<float> searchEmbeddingVector){


    if(fileEmbeddingVector.size() != searchEmbeddingVector.size()) return 0.0f;

    float dotProduct = 0.0f;
    float magA = 0.0f;
    float magB = 0.0f;

    for(size_t i = 0 ; i <fileEmbeddingVector.size(); ++i){
        dotProduct += fileEmbeddingVector[i] * searchEmbeddingVector[i];
        magA += fileEmbeddingVector[i] * fileEmbeddingVector[i];
        magB += searchEmbeddingVector[i] * searchEmbeddingVector[i];
    }

    if (magA == 0.0f || magB == 0.0f) return 0.0f;
    return dotProduct / (std::sqrt(magA) * std::sqrt(magB));

}