/*now we define the functions Creating the sqlite databse and dealing with initializing the databse 
and inserting each files information and vector data*/

#include "DatabaseManager.hpp"
#include <sqlite3.h>
#include <sstream>
#include <iostream>
#include <tuple>

/*First create the initializer that defines the actual path and creates links to the db*/

DatabaseManager::DatabaseManager(const std::string& dbPath){
    //Check if the db exists if not initialize the db 
    if(sqlite3_open(dbPath.c_str(), reinterpret_cast<sqlite3**>(&db)) != SQLITE_OK){
        std::cerr  << "Failed to open database" << sqlite3_errmsg( reinterpret_cast<sqlite3*>(db))<< std::endl;
    }else{
        initializeDatabase();
    }
    
}

DatabaseManager::~DatabaseManager(){
    if(db){
        //closing the sqlite 3 db 
        sqlite3_close(reinterpret_cast<sqlite3*>(db));
    }
}
/*Defining the insert function, Taking in inputs and inserting them into the database*/
void DatabaseManager::initializeDatabase(){
    //creating the sql query to initialize the db 

    const char* createTableSQL =
        "CREATE TABLE IF NOT EXISTS files ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "path TEXT NOT NULL,"
        "name TEXT NOT NULL,"
        "extension TEXT,"
        "embedding TEXT,"
        "last_modified INTEGER"
        ");";

    char* errMsg = nullptr;

    if(sqlite3_exec(reinterpret_cast<sqlite3*>(db), createTableSQL, nullptr, nullptr, &errMsg) != SQLITE_OK){
        std::cerr << "Failed to create DataBase: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }
}
//creating insertion technique 

bool DatabaseManager::insertFile(const std::string& path, const std::string& name, const std::string& extension, const std::vector<float>& embedding, long lastModified){

    if(fileExists(path)){
        std::cout << "Skipping already indexed file: " << name << std::endl;
        if(!fileNeedUpdate(path, lastModified)){
            std::cout << "Skipping unchanged file: " << name << std::endl;
            return false;
        }
        updateFile(path, name, extension, embedding, lastModified);
        std::cout << "Updated existing file: " << name << std::endl;
        return true;
    }
    //serialize the embedding ready for the databse
    std::string serializedEmbedding = embeddingSerializer(embedding);


    //create the command for the sql database
    std::string sql = "INSERT INTO files (path, name, extension, embedding, last_modified) VALUES (?, ?, ?, ?, ?);";

    sqlite3_stmt* stmt;

    //perparing the datebase for insertion 
    if (sqlite3_prepare_v2(reinterpret_cast<sqlite3*>(db), sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare statement." << std::endl;
        return false;
    }


    //insertion statement

    sqlite3_bind_text(stmt, 1, path.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, extension.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 4, serializedEmbedding.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 5, static_cast<sqlite3_int64>(lastModified));

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        std::cerr << "Failed to execute statement." << std::endl;
    }

    sqlite3_finalize(stmt);
    
    return true;
}

std::string DatabaseManager::embeddingSerializer(const std::vector<float>& embedding) {
    std::ostringstream oss;
    for (size_t i = 0; i < embedding.size(); ++i) {
        oss << embedding[i];
        if (i != embedding.size() - 1) {
            oss << ",";
        }
    }
    return oss.str();
}

std::vector<std::tuple<std::string, std::string, std::string, std::vector<float> >> DatabaseManager::getAllFiles(){

    //define the files 
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<float> >> files;


    const char* sql = "SELECT path, name, extension, embedding FROM files;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(reinterpret_cast<sqlite3*>(db), sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare select statement." << std::endl;
        return files;
    }

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        std::string path = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        std::string name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        std::string extension = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        std::string serializedEmbedding = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        
        std::vector<float> embedding = embeddingDeserializer(serializedEmbedding);

        files.push_back({path, name, extension, embedding});
    }

    sqlite3_finalize(stmt);



    return files;
}

std::vector<float> DatabaseManager::embeddingDeserializer(const std::string& embedding){
    std::vector<float> embeddingVector;

    std::stringstream ss(embedding);
        std::string token;
        while (std::getline(ss, token, ',')) {
            embeddingVector.push_back(std::stof(token));
        }

    return embeddingVector;
        
}

bool DatabaseManager::fileExists(const std::string& filePath){

    const char* sql = "SELECT COUNT(*) FROM files WHERE path = ?;";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(reinterpret_cast<sqlite3*>(db), sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare existence check statement." << std::endl;
        return false;
    }

    sqlite3_bind_text(stmt, 1, filePath.c_str(), -1, SQLITE_STATIC);

    int count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int(stmt, 0);
    }

    sqlite3_finalize(stmt);

    return (count > 0);


}

bool DatabaseManager::fileNeedUpdate(const std::string& filePath, long currentModified){

    const char* sql = "SELECT last_modified FROM files WHERE path = ?;";
    sqlite3_stmt* stmt;
    
    if (sqlite3_prepare_v2(reinterpret_cast<sqlite3*>(db), sql, -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare statement for last_modified check." << std::endl;
        return true; // safer to assume needs update
    }

    sqlite3_bind_text(stmt, 1, filePath.c_str(), -1, SQLITE_STATIC);

    long dbModified = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        dbModified = sqlite3_column_int64(stmt, 0);
    }

    sqlite3_finalize(stmt);

    return (currentModified > dbModified); // true if file changed after last index
}

void DatabaseManager::updateFile(const std::string& path, const std::string& name, const std::string& extension, const std::vector<float>& embedding, long lastModified) {
    std::string serializedEmbedding = embeddingSerializer(embedding);

    std::string sql = "UPDATE files SET name = ?, extension = ?, embedding = ?, last_modified = ? WHERE path = ?;";

    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(reinterpret_cast<sqlite3*>(db), sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare update statement." << std::endl;
        return;
    }

    sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 2, extension.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(stmt, 3, serializedEmbedding.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_int64(stmt, 4, static_cast<sqlite3_int64>(lastModified));
    sqlite3_bind_text(stmt, 5, path.c_str(), -1, SQLITE_STATIC);

    if (sqlite3_step(stmt) != SQLITE_DONE) {
        std::cerr << "Failed to execute update." << std::endl;
    }

    sqlite3_finalize(stmt);
}