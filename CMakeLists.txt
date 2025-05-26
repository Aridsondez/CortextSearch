cmake_minimum_required(VERSION 3.10)
project(CortexSearch)


set(CMAKE_CXX_STANDARD 17)

include_directories(include)

# Add ONNX paths manually
include_directories(/opt/homebrew/opt/onnxruntime/include)
link_directories(/opt/homebrew/opt/onnxruntime/lib)

add_executable(CortexSearch
    src/main.cpp
    src/FileScanner.cpp
    src/ContextExtractor.cpp
    src/EmbeddingEngine.cpp
    src/DatabaseManager.cpp
    src/SearchEngine.cpp
)

#Link SQLite 3 
target_link_libraries(CortexSearch sqlite3 onnxruntime)