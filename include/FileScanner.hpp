/*Defining what the Scanner does and defines 
-Name  (Need to be able to extract the name of the file)
-Type (Being able to see the extension of the file is important for the context)
-Path (Path is also important for the location of the file)*/

#pragma once

//The defnition of a file and its parameters are defined with string
#include <string>
#include <vector>

//Define each file itself 

struct FileInfo{
    std::string name;
    std::string extension;
    std::string path;
};

//Now define the FileScanner class
//We won't store any files in the actual class
//instead we will just take in a directory and return the files inside of it

//we have a method which takes in a string and returns the files in the path
class FileScanner{
    public:
        std::vector <FileInfo> scanDirectory(const std::string& directoryPath);
};


