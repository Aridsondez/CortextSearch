// src/DatabaseManager.cpp
// Creates/initializes the SQLite database and handles inserts/updates and reads.
// Stores embeddings as a BLOB (float32[384]) in a separate `embeddings` table.

#include "DatabaseManager.hpp"

#include <sqlite3.h>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>
#include <cstring>   // std::memcpy
#include <cassert>

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────────────────────

DatabaseManager::DatabaseManager(const std::string& dbPath)
    : db(nullptr)
{
    if (sqlite3_open(dbPath.c_str(), &db) != SQLITE_OK) {
        std::cerr << "Failed to open database: " << sqlite3_errmsg(db) << "\n";
        db = nullptr;
        return;
    }

    // Ensure foreign keys are enforced (needed for ON DELETE CASCADE)
    char* err = nullptr;
    if (sqlite3_exec(db, "PRAGMA foreign_keys = ON;", nullptr, nullptr, &err) != SQLITE_OK) {
        std::cerr << "Failed to enable foreign keys: " << (err ? err : "unknown") << "\n";
        sqlite3_free(err);
    }

    initializeDatabase();
}

DatabaseManager::~DatabaseManager() {
    if (db) {
        sqlite3_close(db);
        db = nullptr;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema init (idempotent)
// - Keeps your existing `files` table (including legacy `embedding TEXT` column)
// - Adds `metadata` and `embeddings` tables
// - Records current model configuration
// ─────────────────────────────────────────────────────────────────────────────

void DatabaseManager::initializeDatabase() {
    char* err = nullptr;

    // 1) Ensure files table exists (keeps legacy `embedding TEXT`)
    //    We will STOP using `embedding` column; it remains for backward compatibility.
    const char* createFiles =
        "CREATE TABLE IF NOT EXISTS files ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  path TEXT NOT NULL UNIQUE,"
        "  name TEXT NOT NULL,"
        "  extension TEXT,"
        "  last_modified INTEGER"
        ");";

    if (sqlite3_exec(db, createFiles, nullptr, nullptr, &err) != SQLITE_OK) {
        std::cerr << "Failed to create files: " << err << "\n";
        sqlite3_free(err);
        return;
    }

    // 2) Metadata table (records model info)
    const char* createMeta =
        "CREATE TABLE IF NOT EXISTS metadata ("
        "  key TEXT PRIMARY KEY,"
        "  value TEXT NOT NULL"
        ");";
    if (sqlite3_exec(db, createMeta, nullptr, nullptr, &err) != SQLITE_OK) {
        std::cerr << "Failed to create metadata: " << err << "\n";
        sqlite3_free(err);
        return;
    }

    // 3) New embeddings table (BLOB storage)
    const char* createEmb =
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "  file_id INTEGER PRIMARY KEY,"
        "  vector  BLOB NOT NULL,"
        "  FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE"
        ");";
    if (sqlite3_exec(db, createEmb, nullptr, nullptr, &err) != SQLITE_OK) {
        std::cerr << "Failed to create embeddings: " << err << "\n";
        sqlite3_free(err);
        return;
    }

    // 4) Record current model configuration (idempotent)
    const char* upsertMeta =
        "INSERT OR REPLACE INTO metadata(key, value) VALUES"
        " ('model_name',   'all-MiniLM-L6-v2-ONNX'),"
        " ('embedding_dim','384'),"
        " ('max_seq_len',  '256');";
    if (sqlite3_exec(db, upsertMeta, nullptr, nullptr, &err) != SQLITE_OK) {
        std::cerr << "Failed to upsert metadata: " << err << "\n";
        sqlite3_free(err);
        return;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Insert / Update
// - We upsert the `files` row.
// - We then upsert the `embeddings` row as a BLOB (float32[384]).
// ─────────────────────────────────────────────────────────────────────────────

static bool step_done(sqlite3_stmt* st) {
    int rc = sqlite3_step(st);
    return (rc == SQLITE_DONE);
}

bool DatabaseManager::insertFile(const std::string& path,
                                 const std::string& name,
                                 const std::string& extension,
                                 const std::vector<float>& embedding,
                                 long lastModified)
{
    if (!db) return false;

    // If already present and unchanged → skip
    if (fileExists(path)) {
        if (!fileNeedUpdate(path, lastModified)) {
            std::cout << "Skipping unchanged file: " << name << "\n";
            return false;
        }
        updateFile(path, name, extension, embedding, lastModified);
        std::cout << "Updated existing file: " << name << "\n";
        return true;
    }

    // 1) Upsert into files (insert new row)
    //    NOTE: We DO NOT store the embedding as TEXT anymore.
    const char* upsertFile =
        "INSERT INTO files(path, name, extension, last_modified) "
        "VALUES(?, ?, ?, ?) "
        "ON CONFLICT(path) DO UPDATE SET "
        "  name=excluded.name,"
        "  extension=excluded.extension,"
        "  last_modified=excluded.last_modified;";

    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db, upsertFile, -1, &st, nullptr) != SQLITE_OK) {
        std::cerr << "Failed to prepare upsert files: " << sqlite3_errmsg(db) << "\n";
        return false;
    }
    sqlite3_bind_text(st, 1, path.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 2, name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 3, extension.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, 4, static_cast<sqlite3_int64>(lastModified));
    bool ok = step_done(st);
    sqlite3_finalize(st);
    if (!ok) {
        std::cerr << "Upsert files failed.\n";
        return false;
    }

    // 2) Fetch file_id
    long long file_id = -1;
    const char* selId = "SELECT id FROM files WHERE path=?;";
    if (sqlite3_prepare_v2(db, selId, -1, &st, nullptr) != SQLITE_OK) {
        std::cerr << "Prepare SELECT id failed: " << sqlite3_errmsg(db) << "\n";
        return false;
    }
    sqlite3_bind_text(st, 1, path.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(st) == SQLITE_ROW) file_id = sqlite3_column_int64(st, 0);
    sqlite3_finalize(st);
    if (file_id < 0) {
        std::cerr << "Could not fetch file_id for path: " << path << "\n";
        return false;
    }

    // 3) Upsert embedding as BLOB
    const char* upEmb =
        "INSERT INTO embeddings(file_id, vector) VALUES(?, ?) "
        "ON CONFLICT(file_id) DO UPDATE SET vector=excluded.vector;";
    if (sqlite3_prepare_v2(db, upEmb, -1, &st, nullptr) != SQLITE_OK) {
        std::cerr << "Prepare upsert embeddings failed: " << sqlite3_errmsg(db) << "\n";
        return false;
    }
    sqlite3_bind_int64(st, 1, file_id);
    sqlite3_bind_blob(st, 2, embedding.data(),
                      static_cast<int>(embedding.size() * sizeof(float)), SQLITE_TRANSIENT);
    ok = step_done(st);
    sqlite3_finalize(st);
    if (!ok) {
        std::cerr << "Upsert embedding failed.\n";
        return false;
    }

    return true;
}

void DatabaseManager::updateFile(const std::string& path,
                                 const std::string& name,
                                 const std::string& extension,
                                 const std::vector<float>& embedding,
                                 long lastModified)
{
    if (!db) return;

    // 1) Update file row (no TEXT embedding anymore)
    const char* updFile =
        "UPDATE files SET name=?, extension=?, last_modified=? WHERE path=?;";
    sqlite3_stmt* st=nullptr;
    if (sqlite3_prepare_v2(db, updFile, -1, &st, nullptr) != SQLITE_OK) {
        std::cerr << "Prepare UPDATE files failed: " << sqlite3_errmsg(db) << "\n";
        return;
    }
    sqlite3_bind_text(st, 1, name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(st, 2, extension.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(st, 3, static_cast<sqlite3_int64>(lastModified));
    sqlite3_bind_text(st, 4, path.c_str(), -1, SQLITE_TRANSIENT);
    if (!step_done(st)) std::cerr << "UPDATE files failed.\n";
    sqlite3_finalize(st);

    // 2) Get file_id
    long long file_id = -1;
    const char* selId = "SELECT id FROM files WHERE path=?;";
    if (sqlite3_prepare_v2(db, selId, -1, &st, nullptr) != SQLITE_OK) return;
    sqlite3_bind_text(st, 1, path.c_str(), -1, SQLITE_TRANSIENT);
    if (sqlite3_step(st) == SQLITE_ROW) file_id = sqlite3_column_int64(st, 0);
    sqlite3_finalize(st);
    if (file_id < 0) {
        std::cerr << "Could not fetch file_id for update (path=" << path << ")\n";
        return;
    }

    // 3) Upsert embedding BLOB
    const char* upEmb =
        "INSERT INTO embeddings(file_id, vector) VALUES(?, ?) "
        "ON CONFLICT(file_id) DO UPDATE SET vector=excluded.vector;";
    if (sqlite3_prepare_v2(db, upEmb, -1, &st, nullptr) != SQLITE_OK) {
        std::cerr << "Prepare upsert embeddings failed: " << sqlite3_errmsg(db) << "\n";
        return;
    }
    sqlite3_bind_int64(st, 1, file_id);
    sqlite3_bind_blob(st, 2, embedding.data(),
                      static_cast<int>(embedding.size() * sizeof(float)), SQLITE_TRANSIENT);
    if (!step_done(st)) std::cerr << "Upsert embedding failed.\n";
    sqlite3_finalize(st);
}

// ─────────────────────────────────────────────────────────────────────────────
// Queries / helpers
// ─────────────────────────────────────────────────────────────────────────────

bool DatabaseManager::fileExists(const std::string& filePath) {
    if (!db) return false;

    const char* sql = "SELECT COUNT(*) FROM files WHERE path=?;";
    sqlite3_stmt* st=nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) {
        std::cerr << "Prepare EXISTS failed: " << sqlite3_errmsg(db) << "\n";
        return false;
    }
    sqlite3_bind_text(st, 1, filePath.c_str(), -1, SQLITE_TRANSIENT);

    int count = 0;
    if (sqlite3_step(st) == SQLITE_ROW) count = sqlite3_column_int(st, 0);
    sqlite3_finalize(st);
    return (count > 0);
}

bool DatabaseManager::fileNeedUpdate(const std::string& filePath, long currentModified) {
    if (!db) return true;

    const char* sql = "SELECT last_modified FROM files WHERE path=?;";
    sqlite3_stmt* st=nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) {
        std::cerr << "Prepare last_modified check failed: " << sqlite3_errmsg(db) << "\n";
        return true; // safer to assume needs update
    }
    sqlite3_bind_text(st, 1, filePath.c_str(), -1, SQLITE_TRANSIENT);

    long dbModified = 0;
    if (sqlite3_step(st) == SQLITE_ROW) dbModified = sqlite3_column_int64(st, 0);
    sqlite3_finalize(st);

    return (currentModified > dbModified);
}

// NOTE: This returns rows with the *embedded vector* loaded from the BLOB table.
// If you only need metadata, make a lighter query to avoid pulling blobs.
std::vector<std::tuple<std::string, std::string, std::string, std::vector<float>>>
DatabaseManager::getAllFiles()
{
    std::vector<std::tuple<std::string, std::string, std::string, std::vector<float>>> out;
    if (!db) return out;

    // Join files with embeddings; LEFT JOIN in case some rows are missing vectors.
    const char* sql =
        "SELECT f.path, f.name, f.extension, e.vector "
        "FROM files f LEFT JOIN embeddings e ON e.file_id = f.id;";

    sqlite3_stmt* st=nullptr;
    if (sqlite3_prepare_v2(db, sql, -1, &st, nullptr) != SQLITE_OK) {
        std::cerr << "Prepare SELECT all failed: " << sqlite3_errmsg(db) << "\n";
        return out;
    }

    while (sqlite3_step(st) == SQLITE_ROW) {
        std::string path      = reinterpret_cast<const char*>(sqlite3_column_text(st, 0));
        std::string name      = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
        std::string extension = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));

        std::vector<float> vec;
        const void* blob = sqlite3_column_blob(st, 3);
        int bytes = sqlite3_column_bytes(st, 3);

        if (blob && bytes > 0) {
            // Expect 384 * sizeof(float) = 1536 bytes
            if (bytes % sizeof(float) == 0) {
                size_t n = static_cast<size_t>(bytes / sizeof(float));
                vec.resize(n);
                std::memcpy(vec.data(), blob, static_cast<size_t>(bytes));
            } else {
                std::cerr << "Warning: embedding blob size not a multiple of sizeof(float)\n";
            }
        }

        out.emplace_back(path, name, extension, std::move(vec));
    }

    sqlite3_finalize(st);
    return out;
}

std::vector<FileRow> DatabaseManager::listFiles(int limit){
    std::vector<FileRow> out;
    if (!db) return out;
    
    std::string sql =
        "SELECT id, path, name, extension, last_modified "
        "FROM files "
        "ORDER BY last_modified DESC";
    if (limit > 0) sql += " LIMIT ?";

    sqlite3_stmt* st = nullptr;
    if (sqlite3_prepare_v2(db, sql.c_str(), -1, &st, nullptr) != SQLITE_OK) {
        std::cerr << "Prepare listFiles failed: " << sqlite3_errmsg(db) << "\n";
        return out;
    }
    int bind = 1;
    if (limit > 0) sqlite3_bind_int(st, bind++, limit);

    while (sqlite3_step(st) == SQLITE_ROW) {
        FileRow r;
        r.id            = sqlite3_column_int64(st, 0);
        r.path          = reinterpret_cast<const char*>(sqlite3_column_text(st, 1));
        r.name          = reinterpret_cast<const char*>(sqlite3_column_text(st, 2));
        r.extension     = reinterpret_cast<const char*>(sqlite3_column_text(st, 3));
        r.last_modified = sqlite3_column_int64(st, 4);
        out.emplace_back(std::move(r));
    }
    sqlite3_finalize(st);
    return out;
}