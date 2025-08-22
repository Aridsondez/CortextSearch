#include "TokenizerClient.hpp"
#include <cstdio>
#include <array>
#include <sstream>
#include <nlohmann/json.hpp>
// TokenizerClient class definition puts the pythonEExe, scriptPath, tokenizerJson, and maxLen_ as private members
TokenizerClient::TokenizerClient(std::string pythonExe,
                                 std::string scriptPath,
                                 std::string tokenizerJson,
                                 int maxLen)
: py_(std::move(pythonExe)),
  script_(std::move(scriptPath)),
  tokjson_(std::move(tokenizerJson)),
  maxLen_(maxLen) {}

static std::string sh_escape(const std::string& s){
  std::string out="'";
  for(char c: s){ if(c=='\'') out+="'" "\"'\"" "'"; else out+=c; }
  out+="'"; return out;
}

std::optional<TokenizerResults> TokenizerClient::encode(const std::string& text) const {
  // Build: <venv-python> tools/tokenize.py --tokenizer-json models/tokenizer.json --text "..."
  std::ostringstream cmd;
  cmd << sh_escape(py_) << " "
      << sh_escape(script_) << " --tokenizer-json " << sh_escape(tokjson_)
      << " --text " << sh_escape(text)
      << " --max-len " << maxLen_;

  std::array<char, 4096> buf{};
  std::string out;
  FILE* pipe = popen(cmd.str().c_str(), "r");
  if (!pipe) return std::nullopt;
  while (fgets(buf.data(), (int)buf.size(), pipe)) out += buf.data();
  int rc = pclose(pipe);
  if (rc != 0 || out.empty()) return std::nullopt;

  auto j = nlohmann::json::parse(out, nullptr, false);
  if (j.is_discarded() || !j.contains("input_ids") || !j.contains("attention_mask"))
    return std::nullopt;

   TokenizerResults r;
   r.input_ids      = j["input_ids"].get<std::vector<int64_t>>();
   r.attention_mask = j["attention_mask"].get<std::vector<int64_t>>();
   if (r.input_ids.size() != (size_t)maxLen_ || r.attention_mask.size() != (size_t)maxLen_)
     return std::nullopt;

  return r;
}