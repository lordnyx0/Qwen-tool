from gguf import GGUFReader, GGMLQuantizationType
import numpy as np

r = GGUFReader("modelo_qwen3loop_sft_f16.gguf", mode="r")

# layer_map
lm = r.fields.get("qwen3loop.layer_map")
if lm:
    vals = list(lm.parts[-1])
    print("layer_map (%d entries): %s" % (len(vals), vals[:60]))

# listar todos tensores do bloco 0 e 8
print()
print("Tensores blk.0:", [t.name for t in r.tensors if t.name.startswith("blk.0.")])
print("Tensores blk.8:", [t.name for t in r.tensors if t.name.startswith("blk.8.")])
print("Tensores blk.20:", [t.name for t in r.tensors if t.name.startswith("blk.20.")])
print("Tensores blk.27:", [t.name for t in r.tensors if t.name.startswith("blk.27.")])

# non-blk tensors
nonblk = [t.name for t in r.tensors if not t.name.startswith("blk.")]
print()
print("Non-blk tensors:", nonblk)

# KVs relevantes
for k in ["qwen3loop.block_count", "qwen3loop.context_length"]:
    f = r.fields.get(k)
    if f:
        print("%s = %s" % (k, list(f.parts[-1])))
