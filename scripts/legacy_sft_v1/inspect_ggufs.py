from pathlib import Path
from gguf import GGUFReader

files = [
    "modelo_qwen3loop_sft_f16.gguf",
    "modelo_qwen3loop_sft_q8_0.gguf",
    "unrolled_modelo_qwen3loop_sft_f16.gguf",
    "unrolled_modelo_qwen3loop_sft_q8_0.gguf",
]

for fname in files:
    fpath = Path(fname)
    if not fpath.exists():
        print(f"[-] {fname}: NÃO ENCONTRADO")
        continue

    print(f"\n{'='*60}")
    print(f" ARQUIVO: {fname} ({fpath.stat().st_size/1e9:.2f} GB)")
    print(f"{'='*60}")

    r = GGUFReader(str(fpath), mode="r")
    arch_field = r.fields.get("general.architecture")
    arch = "".join(chr(p) for p in arch_field.parts[-1]).strip("\x00") if arch_field else "desconhecido"
    
    bc_key = f"{arch}.block_count"
    bc = int(r.fields[bc_key].parts[-1][0]) if bc_key in r.fields else "N/A"
    
    blks = set(int(t.name.split(".")[1]) for t in r.tensors if t.name.startswith("blk."))
    has_tokens = "tokenizer.ggml.tokens" in r.fields
    has_merges = "tokenizer.ggml.merges" in r.fields
    
    ft_field = r.fields.get("general.file_type")
    ft = int(ft_field.parts[-1][0]) if ft_field else "N/A"

    print(f"Arquitetura      : {arch}")
    print(f"File Type        : {ft} ({'F16' if ft==1 else 'Q8_0' if ft==8 else ft})")
    print(f"Block Count (KV) : {bc}")
    print(f"Total de Tensores: {len(r.tensors)}")
    print(f"Índices de Blocos: min={min(blks)} max={max(blks)} total_camadas={len(blks)}")
    print(f"Vocabulário      : Tokens={has_tokens} | Merges={has_merges}")
