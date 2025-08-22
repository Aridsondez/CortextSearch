cmake_minimum_required(VERSION 3.10)
project(CortexSearch)


set(CMAKE_CXX_STANDARD 17)

include_directories(include)


# Add ONNX paths manually
include_directories(/opt/homebrew/opt/onnxruntime/include)
include_directories(third_party)  # for third_party/nlohmann/json.hpp (or change if you put it elsewhere)
link_directories(/opt/homebrew/opt/onnxruntime/lib)

add_executable(CortexSearch
    src/main.cpp
    src/FileScanner.cpp
    src/ContextExtractor.cpp
    src/EmbeddingEngine.cpp
    src/DatabaseManager.cpp
    src/SearchEngine.cpp    
    src/TokenizerClient.cpp
)

add_executable(tok_test
    src/tokenizer_smoke.cpp
    src/TokenizerClient.cpp
)
add_executable(embed_test
    src/embed_smoke.cpp
    src/EmbeddingEngine.cpp
    src/TokenizerClient.cpp
    src/ContextExtractor.cpp   # if EmbeddingEngine includes headers from your project that need these
)
target_include_directories(tok_test PRIVATE include third_party)

#Link SQLite 3 
target_link_libraries(CortexSearch sqlite3 onnxruntime)
target_link_libraries(tok_test sqlite3 onnxruntime)
target_link_libraries(embed_test onnxruntime)