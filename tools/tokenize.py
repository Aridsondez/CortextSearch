#!/usr/bin/env python3
import sys, json, argparse
from tokenizers import Tokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-json", required=True)
    ap.add_argument("--text")
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer_json)

    # Try to have the library pad/truncate for us. If not available, we’ll still manually fix length.
    try:
        tok.enable_truncation(max_length=args.max_len)
        tok.enable_padding(length=args.max_len, pad_id=0, pad_token="[PAD]")
    except Exception:
        pass

    def encode_one(text: str):
        enc = tok.encode(text)
        ids  = enc.ids
        mask = getattr(enc, "attention_mask", None)
        if mask is None:
            # If padding is enabled, PAD tokens are 0; build a mask (1 for non‑PAD, 0 for PAD)
            mask = [0 if i == 0 else 1 for i in ids]

        # Fallback: ensure fixed length
        if len(ids) > args.max_len:
            ids, mask = ids[:args.max_len], mask[:args.max_len]
        elif len(ids) < args.max_len:
            pad = args.max_len - len(ids)
            ids  = ids  + [0]*pad
            mask = mask + [0]*pad

        print(json.dumps({"input_ids": ids, "attention_mask": mask}))

    if args.text is not None:
        encode_one(args.text)
    else:
        for line in sys.stdin:
            encode_one(line.rstrip("\n"))

if __name__ == "__main__":
    main()
