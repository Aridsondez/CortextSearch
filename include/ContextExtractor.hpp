/*Alright creating a context extractor that takes in the file path and reads what the file 
is talking about

Things needed
-Function the returns a string once you read inside the file'
-Takes in the file path of the path we are looking at and read the contentss\*/

#pragma once
#include <string>

class ContextExtractor{
    public:
        std::string extractText(const std::string& filePath);

    private:
        std::string extractTxtFile(const std::string& filePath);
        std::string extractPDFFile(const std::string& filePath);
        std::string extractImageFile(const std::string& filePath);
};