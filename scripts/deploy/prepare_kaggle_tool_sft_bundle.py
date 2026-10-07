"""
prepare_kaggle_tool_sft_bundle.py
Atualiza a pasta Kaggle/kaggle_bundle/ com os arquivos necessários para o Treino SFT no Kaggle:
1. Copia data/sft_tool_calling_curated.jsonl para Kaggle/kaggle_bundle/data/
2. Copia scripts/train_sft_tool_qwen3loop.py para Kaggle/kaggle_bundle/train_sft_tool_qwen3loop.py
3. Atualiza dataset-metadata.json para versão do bundle SFT
"""

import os
import shutil
import json

ROOT = os.path.abspath(".")
BUNDLE_DIR = os.path.join(ROOT, "Kaggle", "kaggle_bundle")

print("=== ATUALIZANDO KAGGLE BUNDLE PARA TOOL-CALLING SFT ===")

# 1. Copiar dataset SFT
src_data = os.path.join(ROOT, "data", "sft_tool_calling_curated.jsonl")
dst_data = os.path.join(BUNDLE_DIR, "data", "sft_tool_calling_curated.jsonl")
shutil.copy2(src_data, dst_data)
print(f"Copiado dataset: {dst_data} ({os.path.getsize(dst_data) / 1024 / 1024:.2f} MB)")

# 2. Copiar script de treino SFT
src_script = os.path.join(ROOT, "scripts", "train_sft_tool_qwen3loop.py")
dst_script = os.path.join(BUNDLE_DIR, "train_sft_tool_qwen3loop.py")
shutil.copy2(src_script, dst_script)
print(f"Copiado script de treino: {dst_script}")

# 3. Metadados do dataset
metadata = {
    "title": "qwen3loop-tool-grpo-bundle",
    "id": "lordnyx0/qwen3loop-tool-grpo-bundle",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(os.path.join(BUNDLE_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Kaggle bundle atualizado com sucesso!")
