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




void indexFiles(const std::string& path, DatabaseManager& dbManager, ContextExtractor& extractor, EmbeddingEngine& embedder);
void searchFiles(const std::string& query, DatabaseManager& dbManager, EmbeddingEngine& embedder);
std::time_t getLastModified(std::string& filePath);
bool isCorrectFileType(const std::string extension);

//Using the main to list all the files in the current directory
int main(int argc, char* argv[]){

    //deal with too little arguments
    if(argc < 3){
        std::cout << "Usage:\n";
        std::cout << "  " << argv[0] << " --index <directory_path>\n";
        std::cout << "  " << argv[0] << " --search \"<query>\"\n";
        return 1;
    }

    //deal with processing the arguments 
    std::string mode = argv[1];
    std::string input = argv[2];

    //define all classes used 
    ContextExtractor extractor;
    EmbeddingEngine embedding("../models/model.onnx");
    DatabaseManager manager("cortext.db");

    if (mode == "--index") {
        indexFiles(input, manager, extractor, embedding);
    } else if (mode == "--search") {
        searchFiles(input, manager, embedding);
    } else {
        std::cout << "Unknown mode: " << mode << "\n";
        std::cout << "Usage:\n";
        std::cout << "  " << argv[0] << " --index <directory_path>\n";
        std::cout << "  " << argv[0] << " --search \"<query>\"\n";
        return 1;
    }
    
    return 0;
}


void indexFiles(const std::string& path, DatabaseManager& dbManager, ContextExtractor& extractor, EmbeddingEngine& embedder){
    FileScanner scanner;

    std::vector<FileInfo> files = scanner.scanDirectory(path);

    int indexCount = 0;

    for(const auto& file: files){
        if(isCorrectFileType(file.extension)){

            std::string filePath = file.path;
            
            std::time_t lastModified = getLastModified(filePath);

            std::string context = extractor.extractText(file.path);



            if(context.empty()){
                std::cout << "No text extrated from: " << file.name <<std::endl;
                continue;
            }

            std::vector<float> embeddingVector = embedder.createEmbedding(context);
            

            if(dbManager.insertFile(file.path, file.name, file.extension, embeddingVector, lastModified)){
                std::cout << "Inserted/Updated " << file.path << std::endl;
                ++indexCount;
            }
           

            

        }
    }


    std::cout << "Indexing Completed. Indexed " << indexCount << " New Files" << std::endl;

}


void searchFiles(const std::string& query, DatabaseManager& dbManager, EmbeddingEngine& embedder){
    SearchEngine searcher(dbManager, embedder);

    std::vector<SearchResult> results = searcher.search(query);

    if(results.empty()){
        std::cout<< "No Matching File Found."<< std::endl;
        return;
    }

    std::cout << "\nTop matches:\n";
    for (const auto& result : results) {
        std::cout << "File: " << result.name << " (Score: " << result.score << ")\n";
        std::cout << "Path: " << result.path << "\n";
        std::cout << "--------------------------------------\n";
    }

}


bool isCorrectFileType(const std::string extension){

    if (extension == ".txt" || extension == ".pdf" || extension == ".png" 
        || extension == ".jpg" || extension == ".jpeg"){
            return true;
    }

    return false;
}

std::time_t getLastModified(std::string& filePath){
    auto ftime = std::filesystem::last_write_time(filePath);
        
        auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
            ftime - std::filesystem::file_time_type::clock::now()
            + std::chrono::system_clock::now()
        );

        std::time_t lastModified = std::chrono::system_clock::to_time_t(sctp);
    return lastModified;
}