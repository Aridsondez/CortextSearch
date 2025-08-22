#include "TokenizerClient.hpp"
#include <iostream>

int main() {
  // Use your venv’s Python so the helper can import `tokenizers`
  TokenizerClient tok{
    "./.venv/bin/python",     // python interpreter inside your venv
    "tools/tokenize.py",      // helper script
    "models/tokenizer.json",  // tokenizer spec
    256
  };

  auto r = tok.encode("hello world");
  if (!r) { std::cerr << "encode failed\n"; return 1; }

  std::cout << "ids[0..6]: ";
  for (int i = 0; i < 7; ++i) std::cout << r->input_ids[i] << " ";
  std::cout << "\nmask[0..6]: ";
  for (int i = 0; i < 7; ++i) std::cout << r->attention_mask[i] << " ";
  std::cout << "\nlen=" << r->input_ids.size() << "\n";
}
