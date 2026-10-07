"""
prepare_kaggle_bundle.py
Cria e organiza a pasta Kaggle/kaggle_bundle/ contendo tudo o que é necessário para o treino no Kaggle:
- Modelo base (checkpoints/qwen3loop_sft_final)
- Currículo (data/rl_curriculum)
- Engine customizada (engine/qwen3loop_python)
- Recompensas (scripts/rewards_tool_calling.py)
- Script de treino Dual T4 (train_dual_t4_grpo.py)
- Notebook Jupyter autocontido (qwen3loop_dual_t4_grpo.ipynb)
- dataset-metadata.json para upload direto via Kaggle API
"""

import os
import shutil
import json

ROOT = os.path.abspath(".")
BUNDLE_DIR = os.path.join(ROOT, "Kaggle", "kaggle_bundle")

print("=== CRIANDO KAGGLE BUNDLE ===")
if os.path.exists(BUNDLE_DIR):
    shutil.rmtree(BUNDLE_DIR)
os.makedirs(BUNDLE_DIR, exist_ok=True)

# 1. Copiar modelo base
dst_model = os.path.join(BUNDLE_DIR, "model")
print("1. Copiando modelo base...")
shutil.copytree(os.path.join(ROOT, "checkpoints", "qwen3loop_sft_final"), dst_model)

# 2. Copiar dados de currículo
dst_data = os.path.join(BUNDLE_DIR, "data")
print("2. Copiando dataset de currículo...")
shutil.copytree(os.path.join(ROOT, "data", "rl_curriculum"), dst_data)

# 3. Copiar engine do modelo qwen3loop
dst_engine = os.path.join(BUNDLE_DIR, "engine")
print("3. Copiando engine customizada do qwen3loop...")
shutil.copytree(os.path.join(ROOT, "engine", "qwen3loop_python"), os.path.join(dst_engine, "qwen3loop_python"))

# 4. Copiar rewards_tool_calling.py
print("4. Copiando rewards_tool_calling.py...")
shutil.copy2(os.path.join(ROOT, "scripts", "rewards_tool_calling.py"), os.path.join(BUNDLE_DIR, "rewards_tool_calling.py"))

# 5. Criar dataset-metadata.json
metadata = {
    "title": "qwen3loop-tool-grpo-bundle",
    "id": "lordnyx0/qwen3loop-tool-grpo-bundle",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(os.path.join(BUNDLE_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nBundle de arquivos montado com sucesso em:", BUNDLE_DIR)
