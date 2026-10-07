from gguf import GGUFReader
r = GGUFReader("modelo_qwen3loop_sft_f16.gguf", mode="r")
lm_field = r.fields.get("qwen3loop.layer_map")
print("layer_map field types:", lm_field.types)
print("layer_map parts count:", len(lm_field.parts))
for i, p in enumerate(lm_field.parts):
    print(f"  part[{i}]: dtype={p.dtype} len={len(p)} vals={list(p)[:10]}")
