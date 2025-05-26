#include "ContextExtractor.hpp"
//Include fstream and sstream handles file inputs and output
//fstream handles files and sstream handles string inputs

#include <fstream>
#include <sstream>
#include <filesystem>
#include <cstdlib>  // for system()
#include <iostream>


std::string ContextExtractor::extractText(const std::string& filePath){
    std::string extension = std::filesystem::path(filePath).extension().string();

    if(extension == ".txt"){
        return extractTxtFile(filePath);
    }else if(extension == ".pdf"){
        return extractPDFFile(filePath);
    }else if(extension == ".jpg" || extension ==".png" || extension == "jpeg"){
        return extractImageFile(filePath);
    }
    return "";
}

std::string ContextExtractor::extractTxtFile(const std::string& filePath){
    std::ifstream file(filePath);

    if(!file.is_open()){
        return "";
    }

    std::stringstream context;
    context << file.rdbuf();
    return context.str();
}

std::string ContextExtractor::extractPDFFile(const std::string& filePath){
    //deal with extracting the pdf file into plain text
    //running the command to convert to pdf to straight text to extract meaning
    std::string command = "pdftotext \"" + filePath + "\" -";
    std::string result;
    char buffer[128];

    //reading the pdf file
    FILE* pipe = popen(command.c_str(), "r");
    if(!pipe) return "";


    //reading in the file by the buffer size till we get to the end of the file
    while(fgets(buffer, sizeof(buffer), pipe)!= nullptr){
        result +=buffer;
    }

    pclose(pipe);

    return result;

}

std::string ContextExtractor::extractImageFile(const std::string& filePath){

    std::string command = "tesseract \"" + filePath +"\" stdout";

    std::string result;

    char buffer[128];

    FILE* pipe = popen(command.c_str(), "r");
    if(!pipe) return "";


    //reading in the file by the buffer size till we get to the end of the file
    while(fgets(buffer, sizeof(buffer), pipe)!= nullptr){
        result +=buffer;
    }

    pclose(pipe);
    return result;
}