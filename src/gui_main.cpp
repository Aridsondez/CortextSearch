// src/gui_main.cpp
#include "FileScanner.hpp"
#include "ContextExtractor.hpp"
#include "EmbeddingEngine.hpp"
#include "DatabaseManager.hpp"
#include "SearchEngine.hpp"

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "misc/cpp/imgui_stdlib.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

// ---------- helpers copied from CLI semantics ----------
static bool isCorrectFileType(const std::string& extension) {
    return (extension == ".txt" || extension == ".pdf" || extension == ".png" ||
            extension == ".jpg" || extension == ".jpeg");
}

static std::time_t getLastModified(std::string& filePath) {
    auto ftime = std::filesystem::last_write_time(filePath);
    auto sctp = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
        ftime - std::filesystem::file_time_type::clock::now()
        + std::chrono::system_clock::now()
    );
    return std::chrono::system_clock::to_time_t(sctp);
}

// tiny macOS opener
static void openFileNative(const std::string& path) {
#ifdef __APPLE__
    std::string cmd = "open \"" + path + "\"";
    std::system(cmd.c_str());
#endif
}

int main() {
    // ------------- GLFW / OpenGL -------------
    if (!glfwInit()) return -1;
    const char* glsl_version = "#version 150";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    GLFWwindow* window = glfwCreateWindow(1280, 800, "CortexSearch", nullptr, nullptr);
    if (!window) { glfwTerminate(); return -1; }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    // ------------- ImGui -------------
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // ------------- Backend objects (same as CLI main.cpp) -------------
    const std::string onnxModelPath   = "models/model.onnx";
    const std::string pythonExe       = "./.venv/bin/python";
    const std::string tokenizerScript = "tools/tokenize.py";
    const std::string tokenizerJson   = "models/tokenizer.json";
    const size_t      maxSeqLen       = 256;

    ContextExtractor extractor;
    EmbeddingEngine  embedder(onnxModelPath, pythonExe, tokenizerScript, tokenizerJson, maxSeqLen);
    DatabaseManager  db("cortex.db");
    SearchEngine     searcher(db, embedder);
    FileScanner      scanner;

    // ------------- UI state -------------
    std::vector<FileRow> indexedFiles;   // <— now FileRow, not tuples
    std::string fileFilter;
    std::string queryText;
    std::vector<SearchResult> searchResults;

    std::string indexDirPath;
    std::atomic<bool> isIndexing{false};
    std::atomic<int> filesIndexed{0};
    std::atomic<int> filesDiscovered{0};
    std::string indexStatus;

    // show currently indexed files (simple and robust):
    // reuse your existing heavy reader for now (loads blobs but works).
    // If you add a light listFiles() later, swap it in here.
    auto loadIndexedFiles = [&]() {
        // returns tuples: (path, name, extension, vector<float>)
        return db.listFiles(200);
    };
    indexedFiles = loadIndexedFiles();

    auto refreshIndexedFiles = [&]() {
        indexedFiles = loadIndexedFiles();
    };

    auto startIndexing = [&]() {
        if (isIndexing.load()) return;
        if (indexDirPath.empty()) { indexStatus = "Choose a directory first."; return; }

        isIndexing = true;
        filesIndexed = 0;
        filesDiscovered = 0;
        indexStatus = "Indexing…";

        std::thread([&, dir = indexDirPath](){
            try {
                // 1) discover files exactly like CLI
                std::vector<FileInfo> files = scanner.scanDirectory(dir);
                filesDiscovered = static_cast<int>(files.size());

                // 2) process each file (same flow as CLI indexFiles)
                int localCount = 0;
                for (const auto& file : files) {
                    if (!isCorrectFileType(file.extension)) continue;

                    std::string filePath = file.path;
                    std::time_t lastModified = getLastModified(filePath);

                    std::string context = extractor.extractText(file.path);
                    if (context.empty()) continue;

                    std::vector<float> embeddingVector = embedder.createEmbedding(context);
                    if (embeddingVector.empty()) continue;

                    if (db.insertFile(file.path, file.name, file.extension, embeddingVector, lastModified)) {
                        ++localCount;
                        filesIndexed = localCount;
                    }
                }
                indexStatus = "Index complete.";
            } catch (const std::exception& e) {
                indexStatus = std::string("Index error: ") + e.what();
            }
            isIndexing = false;
            refreshIndexedFiles();
        }).detach();
    };

    // ------------- Main loop -------------
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // Left pane: Indexed Files
        ImGui::Begin("Indexed Files");
        if (ImGui::Button("Refresh")) {
            refreshIndexedFiles();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("%zu files", indexedFiles.size());

        ImGui::InputTextWithHint("##filter", "Filter by name/path…", &fileFilter);

        if (ImGui::BeginTable("filesTable", 2, ImGuiTableFlags_RowBg|ImGuiTableFlags_BordersV|ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch, 0.5f);
            ImGui::TableSetupColumn("Path", ImGuiTableColumnFlags_WidthStretch, 1.5f);
            ImGui::TableHeadersRow();

            for (const auto& f : indexedFiles) {
                if (!fileFilter.empty()) {
                    std::string hay = f.name + " " + f.path;
                    if (hay.find(fileFilter) == std::string::npos) continue;
                }
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                if (ImGui::Selectable(f.name.c_str(), false)) {}
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                    openFileNative(f.path);
                }
                ImGui::TableSetColumnIndex(1);
                ImGui::TextUnformatted(f.path.c_str());
                        }
                        ImGui::EndTable();
        }
        ImGui::End();

        // Right pane: Search + Index controls
        ImGui::Begin("Search & Index");

        ImGui::SeparatorText("Search");
        bool doSearch = false;
        if (ImGui::InputTextWithHint("##query", "Type your query… (Enter to search)", &queryText,
                                     ImGuiInputTextFlags_EnterReturnsTrue)) {
            doSearch = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Search")) doSearch = true;

        if (doSearch) {
            // Use your CLI's exact path: SearchEngine::search(query)
            searchResults = searcher.search(queryText);
        }

        if (ImGui::BeginTable("resultsTable", 3, ImGuiTableFlags_RowBg|ImGuiTableFlags_BordersV|ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 80.f);
            ImGui::TableSetupColumn("Name",  ImGuiTableColumnFlags_WidthStretch, 0.6f);
            ImGui::TableSetupColumn("Path",  ImGuiTableColumnFlags_WidthStretch, 1.4f);
            ImGui::TableHeadersRow();

            for (const auto& r : searchResults) {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("%.3f", r.score);

                ImGui::TableSetColumnIndex(1);
                if (ImGui::Selectable(r.name.c_str(), false)) {}
                if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0)) {
                    openFileNative(r.path);
                }

                ImGui::TableSetColumnIndex(2);
                ImGui::TextUnformatted(r.path.c_str());
            }
            ImGui::EndTable();
        }

        ImGui::SeparatorText("Index Directory");
        ImGui::InputTextWithHint("##indexdir", "/path/to/dir…", &indexDirPath);
        ImGui::SameLine();
        bool canIndex = !isIndexing.load() && !indexDirPath.empty();
        if (!canIndex) ImGui::BeginDisabled();
        if (ImGui::Button("Index")) startIndexing();
        if (!canIndex) ImGui::EndDisabled();

        if (isIndexing.load()) {
            float progress = 0.0f;
            int total = filesDiscovered.load(); // we set this after discovering
            int done  = filesIndexed.load();
            if (total > 0) progress = float(done) / float(total);
            ImGui::ProgressBar(progress, ImVec2(-1.0f, 0.0f),
                               (std::to_string(done) + "/" + std::to_string(total)).c_str());
        }
        ImGui::TextUnformatted(indexStatus.c_str());

        ImGui::End();

        // Render
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
