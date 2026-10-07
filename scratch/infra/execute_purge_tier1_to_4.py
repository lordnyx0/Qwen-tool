import os
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')

def get_size(path):
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path) or os.path.islink(path):
        return os.path.getsize(path)
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if not os.path.islink(fp):
                try:
                    total += os.path.getsize(fp)
                except Exception:
                    pass
    return total

def fmt(b):
    return f"{b / (1024**3):.2f} GB" if b >= 1024**3 else f"{b / (1024**2):.2f} MB"

def safe_remove(path):
    if not os.path.exists(path):
        return 0
    size = get_size(path)
    try:
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
        print(f"  [REMOVIDO] {path} ({fmt(size)})")
        return size
    except Exception as e:
        print(f"  [ERRO AO REMOVER] {path}: {e}")
        return 0

targets = {
    "TIER 1 (Duplicatas pesadas, zips antigos e clones)": [
        "checkpoints/qwen3loop_tool_sft_v18",
        "checkpoints/qwen3loop_tool_sft_download",
        "kaggle_dataset_upload/kaggle_bundle.zip",
        "kaggle_output",
        "kaggle_bundle/model",
        "model_src/model.safetensors"
    ],
    "TIER 2A (GGUFs obsoletos do step intermediário ckpt-150)": [
        "models/gguf/modelo_qwen3loop_tool_sft_ckpt150_f16.gguf",
        "models/gguf/unrolled_modelo_qwen3loop_tool_sft_ckpt150_f16.gguf",
        "models/gguf/unrolled_modelo_qwen3loop_tool_sft_ckpt150_q8_0.gguf"
    ],
    "TIER 2B (GGUFs F16 intermediários substituídos por Q8_0)": [
        "models/gguf/modelo_qwen3loop_grpo_f16.gguf",
        "models/gguf/modelo_qwen3loop_sft_f16.gguf",
        "models/gguf/modelo_qwen3loop_tool_sft_f16.gguf",
        "models/gguf/unrolled_modelo_qwen3loop_grpo_f16.gguf",
        "models/gguf/unrolled_modelo_qwen3loop_sft_f16.gguf",
        "models/gguf/unrolled_modelo_qwen3loop_tool_sft_f16.gguf"
    ],
    "TIER 3 (Checkpoints intermediários antigos - steps 25..300)": [
        "checkpoints/qwen3loop_tool_sft_final/checkpoint-150",
        "checkpoints/qwen3loop_tool_sft_final/checkpoint-300",
        "checkpoints/qwen3loop_tool_grpo_final/checkpoint-25",
        "checkpoints/qwen3loop_tool_grpo_final/checkpoint-50",
        "checkpoints/qwen3loop_tool_grpo_final/checkpoint-75",
        "checkpoints/qwen3loop_tool_grpo_final/checkpoint-100"
    ],
    "TIER 4 (Datasets legados anteriores ao Curriculum 3000)": [
        "data/candidates_raw_25k.jsonl",
        "data/sft_annotated_curated.jsonl",
        "data/sft_curated_11k.jsonl",
        "data/sft_agentic_code_mix.jsonl",
        "data/sft_tool_calling_curated.jsonl",
        "data/sft_train_sample_200.jsonl"
    ]
}

print("=" * 80)
print("INICIANDO EXECUÇÃO DA PURGA TOTAL RECOMENDADA")
print("=" * 80)

total_freed = 0
for tier_name, item_list in targets.items():
    print(f"\n--- {tier_name} ---")
    tier_freed = 0
    for item in item_list:
        p = os.path.abspath(item)
        if os.path.exists(p):
            tier_freed += safe_remove(p)
        else:
            print(f"  [IGNORADO - JÁ NÃO EXISTE] {item}")
    print(f"Subtotal liberado em {tier_name.split(' (')[0]}: {fmt(tier_freed)}")
    total_freed += tier_freed

print("\n" + "=" * 80)
print(f"PURGA CONCLUÍDA COM SUCESSO!")
print(f"ESPAÇO TOTAL LIBERADO EM DISCO: {fmt(total_freed)}")
print("=" * 80)
