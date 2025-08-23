/*Creating the databse manager. Think about what you would need to manage in the database
--Handling the Schema of the database initialzing the structure
--Dealing with adding to the db
--Dealing with removing 
--Dealing with deleting the whole db 
--initializing the db(so we are using sqlite which exists on the local server so just getting the path)*/

#pragma once
#include <string>
#include <vector>
#include <cstdlib>
#include <sqlite3.h>
#include <tuple>

struct FileRow{
    long long id;
    std::string path;
    std::string name;
    std::string extension;
    long long last_modified;
};

class DatabaseManager{
    public:
        DatabaseManager(const std::string& dbPath);
        ~DatabaseManager();


        //dealing with insertion, have to see what information about the file we are inserting
        bool insertFile(const std::string& path, const std::string& name, 
            const std::string& extension, const std::vector<float>& embedding, long lastModified);
        
        //dealing with other operations 
        void updateFile(const std::string& path, const std::string& name, const std::string& extension, const std::vector<float>& embedding, long lastModified);

        //getting all files from db
        std::vector<std::tuple<std::string, std::string, std::string, std::vector<float> >> getAllFiles();

        std::vector<FileRow> listFiles(int limit=200);


    private:
        sqlite3* db;
        //basically changing the information into something that can be stored in the db
        //so the vectors that I have being a string of vectors has to be serialized for the db
        void initializeDatabase();
        //Also need a reference to the actual databse 

        //Protects against duplicates 
        bool fileExists(const std::string& filePath);

        //checking last modified date
        bool fileNeedUpdate(const std::string& filePath, long currentModified);
};