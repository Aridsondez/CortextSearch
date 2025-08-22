/*Now generate an embedding based on the actual inputs of the file contents*/

#include "EmbeddingEngine.hpp"
#include <iostream>
#include <vector>


//opening up the downloaded ai to access the model and create embeddings using such model
//currently maybe not working because i can't access the actuall library that allows me to access the model

EmbeddingEngine::EmbeddingEngine(const std::string& onnxModelPath,
                                 const std::string& pythonExe,
                                 const std::string& tokenizerScript,
                                 const std::string& tokenizerJson,
                                 size_t maxSeqLen)
: env(ORT_LOGGING_LEVEL_WARNING, "embed"),
  session(nullptr),
  maxSeqLen_(maxSeqLen)
{
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session = Ort::Session(env, onnxModelPath.c_str(), sessionOptions);

    // cache input/output names (we’ll match by substring)
    Ort::AllocatorWithDefaultOptions alloc;
    size_t ni = session.GetInputCount();
    size_t no = session.GetOutputCount();

    inputNamesOwned_.reserve(ni);
    outputNamesOwned_.reserve(no);
    for (size_t i = 0; i < ni; ++i) {
        Ort::AllocatedStringPtr name = session.GetInputNameAllocated(i, alloc);
        inputNamesOwned_.emplace_back(name.get()); // copy into std::string
    }
    for (size_t i = 0; i < no; ++i) {
        Ort::AllocatedStringPtr name = session.GetOutputNameAllocated(i, alloc);
        outputNamesOwned_.emplace_back(name.get());
    }
    // create the tokenizer bridge (python helper)
    tok_ = std::make_unique<TokenizerClient>(pythonExe, tokenizerScript, tokenizerJson, (int)maxSeqLen_);

    // (optional) log names once
    std::cerr << "[ONNX] Inputs:";
    for (auto n : inputNamesOwned_) std::cerr << " " << n;
    std::cerr << "\n[ONNX] Outputs:";
    for (auto n : outputNamesOwned_) std::cerr << " " << n;
    std::cerr << std::endl;
}

std::vector<float> EmbeddingEngine::createEmbedding(const std::string& text) {
    if (!tok_) return {};

    // 1) tokenize
    auto T = tok_->encode(text);
    if (!T) return {};

    const int64_t seq = (int64_t)T->input_ids.size(); // 256
    // 2) build ONNX tensors
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> shape{1, seq};

    std::vector<int64_t> input_ids = T->input_ids;
    std::vector<int64_t> attention_mask = T->attention_mask;
    std::vector<int64_t> token_type_ids(seq, 0);

    Ort::Value idsT  = Ort::Value::CreateTensor<int64_t>(mem, input_ids.data(), input_ids.size(), shape.data(), shape.size());
    Ort::Value masT  = Ort::Value::CreateTensor<int64_t>(mem, attention_mask.data(), attention_mask.size(), shape.data(), shape.size());
    Ort::Value ttiT  = Ort::Value::CreateTensor<int64_t>(mem, token_type_ids.data(), token_type_ids.size(), shape.data(), shape.size());

    // 3) map names -> tensors (robust to different ordering)
    std::vector<const char*> inNames;
    std::vector<Ort::Value>  inVals;
    for (auto& n : inputNamesOwned_) {
        if (n.find("input_ids") != std::string::npos) {
            inNames.push_back(n.c_str());
            inVals.emplace_back(std::move(idsT));
        } else if (n.find("attention_mask") != std::string::npos) {
            inNames.push_back(n.c_str());
            inVals.emplace_back(std::move(masT));
        } else if (n.find("token_type_ids") != std::string::npos) {
            inNames.push_back(n.c_str());
            inVals.emplace_back(std::move(ttiT));
        }
    }   
    std::vector<const char*> outNames;
    outNames.reserve(outputNamesOwned_.size());
    for (auto& n : outputNamesOwned_) outNames.push_back(n.c_str());

    // 4) run
    auto outs = session.Run(Ort::RunOptions{nullptr},
                            inNames.data(), inVals.data(), inVals.size(),
                            outNames.data(), outNames.size());

    // 5) expect last_hidden_state [1, seq, 384] → masked mean‑pool
    auto& out = outs[0];
    auto info = out.GetTensorTypeAndShapeInfo();
    auto shp  = info.GetShape(); // [1, seq, hidden]
    if (shp.size() != 3 || shp[0] != 1) return {};
    const int64_t hidden = shp[2];

    const float* H = out.GetTensorData<float>(); // size seq*hidden
    std::vector<float> pooled(hidden, 0.0f);
    double denom = 0.0;

    for (int64_t t = 0; t < seq; ++t) {
        if (attention_mask[t] == 0) continue;
        const float* row = H + t * hidden;
        for (int64_t h = 0; h < hidden; ++h) pooled[h] += row[h];
        denom += 1.0;
    }
    if (denom < 1e-6) denom = 1.0;
    for (int64_t h = 0; h < hidden; ++h) pooled[h] = static_cast<float>(pooled[h] / denom);

    // 6) L2‑normalize
    double norm = 0.0;
    for (float v : pooled) norm += double(v) * double(v);
    norm = std::sqrt(std::max(norm, 1e-12));
    for (auto& v : pooled) v = float(v / norm);

    return pooled; // length should be 384
}