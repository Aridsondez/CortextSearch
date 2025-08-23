cmake_minimum_required(VERSION 3.10)
project(CortexSearch)

# ---------------------------
# Toolchain / language
# ---------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Silence macOS OpenGL deprecation warnings (safe no-op elsewhere)
if(APPLE)
  add_definitions(-DGL_SILENCE_DEPRECATION)
endif()

# ---------------------------
# Include & Link paths (UNCHANGED)
# ---------------------------
include_directories(include)
include_directories(/opt/homebrew/opt/onnxruntime/include)
include_directories(third_party)  # for third_party/nlohmann/json.hpp (or adjust)
link_directories(/opt/homebrew/opt/onnxruntime/lib)

# ---------------------------
# Core source list (NEW)
# We reuse these for both CLI and GUI so we don't duplicate lists.
# ---------------------------
set(CORE_SOURCES
    src/FileScanner.cpp
    src/ContextExtractor.cpp
    src/EmbeddingEngine.cpp
    src/DatabaseManager.cpp
    src/SearchEngine.cpp
    src/TokenizerClient.cpp
)

# ---------------------------
# CLI executable (UNCHANGED behavior; just uses CORE_SOURCES var)
# ---------------------------
add_executable(CortexSearch
    src/main.cpp
    ${CORE_SOURCES}
)

# ---------------------------
# Tokenizer/Embed smoke tests (UNCHANGED)
# ---------------------------
add_executable(tok_test
    src/tokenizer_smoke.cpp
    src/TokenizerClient.cpp
)
add_executable(embed_test
    src/embed_smoke.cpp
    src/EmbeddingEngine.cpp
    src/TokenizerClient.cpp
    src/ContextExtractor.cpp
)

target_include_directories(tok_test PRIVATE include third_party)

# ---------------------------
# Link existing deps (UNCHANGED)
# ---------------------------
target_link_libraries(CortexSearch sqlite3 onnxruntime)
target_link_libraries(tok_test sqlite3 onnxruntime)
target_link_libraries(embed_test onnxruntime)

# =================================================================
#                  GUI: Dear ImGui + GLFW + OpenGL  (NEW)
# =================================================================

# 1) Find OpenGL
find_package(OpenGL REQUIRED)

# 2) Find GLFW (Homebrew provides a CMake config named glfw3)
#    If your system only has "glfw", try swapping the lines.
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
  find_package(glfw REQUIRED)           # fallback if package is named "glfw"
endif()
# Determine the GLFW target name and get its include dirs
if(TARGET glfw)
  set(GLFW_LIB_TARGET glfw)
  get_target_property(GLFW_INCLUDE_DIRS glfw INTERFACE_INCLUDE_DIRECTORIES)
elseif(TARGET glfw3)
  set(GLFW_LIB_TARGET glfw3)
  get_target_property(GLFW_INCLUDE_DIRS glfw3 INTERFACE_INCLUDE_DIRECTORIES)
else()
  message(FATAL_ERROR "Could not find a GLFW CMake target (glfw or glfw3).")
endif()

if(NOT GLFW_INCLUDE_DIRS)
  set(GLFW_INCLUDE_DIRS /opt/homebrew/include)
endif()
# 3) Dear ImGui from sources
#    We compile ImGui, its GLFW/OpenGL3 backends, and the bundled gl3w loader.
add_library(imgui STATIC
  external/imgui/imgui.cpp
  external/imgui/imgui_draw.cpp
  external/imgui/imgui_tables.cpp
  external/imgui/imgui_widgets.cpp
  external/imgui/misc/cpp/imgui_stdlib.cpp
  external/imgui/backends/imgui_impl_glfw.cpp
  external/imgui/backends/imgui_impl_opengl3.cpp
)

target_include_directories(imgui PUBLIC
  external/imgui
  external/imgui/backends
)
# Ensure ImGui sees GLFW headers during its own compilation
target_include_directories(imgui PRIVATE ${GLFW_INCLUDE_DIRS})

# Use GLEW for the OpenGL loader
find_package(GLEW REQUIRED)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)
target_link_libraries(imgui PUBLIC GLEW::GLEW ${GLFW_LIB_TARGET})

# 4) GUI executable
#    This is your new windowed app with ImGui. It reuses all core sources
#    but has a different main() in src/gui_main.cpp.
add_executable(CortexSearchGUI
    src/gui_main.cpp
    ${CORE_SOURCES}
)

# Make sure core headers/third_party headers are visible to GUI too
target_include_directories(CortexSearchGUI PRIVATE include third_party)

# Link GUI deps
# - On some macOS setups, GLFW needs explicit Apple frameworks; add if you see link errors.
target_link_libraries(CortexSearchGUI
    PRIVATE
    imgui
    OpenGL::GL
    ${GLFW_LIB_TARGET}
    sqlite3
    onnxruntime
)

if(APPLE)
  # Only add these if you hit unresolved symbols for Cocoa/IOKit/CoreVideo.
  target_link_libraries(CortexSearchGUI PRIVATE "-framework Cocoa" "-framework IOKit" "-framework CoreVideo")
endif()

# =================================================================
#                          End GUI section
# =================================================================
