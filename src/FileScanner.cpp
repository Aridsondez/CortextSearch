//Creating the actual file Scanner file 

//including the file scanner and the file system 
//Filesystem gives a standerdized way of interacting with the files using paths
#include "FileScanner.hpp"
#include <filesystem>

namespace fs = std::filesystem;

std::vector<FileInfo> FileScanner::scanDirectory(const std::string& directoryPath){
    //define the array for files 
    std::vector<FileInfo> files;


    //looping through all the files in the directory and checking if its a regular file
    for(const auto&entry : fs::recursive_directory_iterator(directoryPath)){
        if(entry.is_regular_file()){
            FileInfo info;

            fs::path normalizedPath = fs::absolute(entry.path()).lexically_normal();

            info.path = normalizedPath.string();
            info.name = entry.path().filename().string();
            info.extension = entry.path().extension().string();
            files.push_back(info);
        }
    }

    return files;
}