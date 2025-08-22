import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
import sys

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "model.onnx"
TOK = ROOT / "models" / "tokenizer.json"

# 1) Load tokenizer and encode text
text = "hello world"
tok = Tokenizer.from_file(str(TOK))
max_len = 256

try:
    tok.enable_truncation(max_length=max_len)
    tok.enable_padding(length=max_len, pad_id=0, pad_token="[PAD]")
except Exception:
    pass

enc = tok.encode(text)
ids = enc.ids
mask = getattr(enc, "attention_mask", None) or [1]*len(ids)
# ensure fixed length
if len(ids) > max_len:
    ids, mask = ids[:max_len], mask[:max_len]
elif len(ids) < max_len:
    pad = max_len - len(ids)
    ids  = ids  + [0]*pad
    mask = mask + [0]*pad

ids  = np.array([ids], dtype=np.int64)   # [1, 256]
mask = np.array([mask], dtype=np.int64)  # [1, 256]
ttids = np.zeros_like(mask, dtype=np.int64)  # token_type_ids all zeros

# 2) Load ONNX
sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])

# 3) Build feeds by matching names (robust across exports)
feeds = {}
for i in sess.get_inputs():
    n = i.name
    if "input_ids" in n:
        feeds[n] = ids
    elif "attention_mask" in n:
        feeds[n] = mask
    elif "token_type_ids" in n:
        feeds[n] = ttids

# 4) Run
outs = sess.run(None, feeds)

# 5) Print shapes and a tiny preview
shapes = [np.array(o).shape for o in outs]
print("Output shapes:", shapes)

# If we get last_hidden_state [1, 256, 384], do mean-pooling here just to confirm:
emb = None
for idx, o in enumerate(outs):
    arr = np.array(o)
    if arr.ndim == 3 and arr.shape[2] == 384:
        # masked mean pool over seq
        m = mask.astype(np.float32)
        denom = np.clip(m.sum(axis=1, keepdims=True), 1.0, None)  # avoid div/0
        emb = (arr * m[..., None]).sum(axis=1) / denom
        break
    if arr.ndim == 2 and arr.shape[1] == 384:
        emb = arr
        break

if emb is None:
    print("Could not find a 384-d embedding in outputs; check model/export.", file=sys.stderr)
    sys.exit(2)

# L2 normalize
norm = np.linalg.norm(emb, axis=1, keepdims=True)
norm[norm == 0] = 1.0
emb = emb / norm

print("Embedding shape:", emb.shape)        # expect (1, 384)
print("Embedding[0][:8]:", emb[0][:8])      # small peek
