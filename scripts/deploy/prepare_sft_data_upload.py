"""
prepare_sft_data_upload.py
Prepara o bundle leve do Kaggle (kaggle_sft_data/) contendo:
- data/sft_tool_calling_v2.jsonl (3.625 amostras de elite)
- scripts/train_sft_tool_qwen3loop.py (com suporte a DDP, multi-exit e probe supervision)
- engine/qwen3loop_python/ (código da arquitetura)
- data/tool_catalog/tools_schema.py (catálogo das 50 ferramentas)
- data/eval_benchmark_battery_v2.jsonl (bateria de 121 itens selada)
- dataset-metadata.json (para envio via Kaggle CLI)
"""

import os
import shutil
import json

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.join(ROOT, "kaggle_sft_data")

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print("PREPARANDO BUNDLE KAGGLE SFT v2 (kaggle_sft_data/)")
print("=" * 70)

# 1. Copiar dataset SFT v2 (salvando com ambos os nomes para compatibilidade)
src_data = os.path.join(ROOT, "data", "sft_tool_calling_v2.jsonl")
dst_data_v2 = os.path.join(OUT_DIR, "sft_tool_calling_v2.jsonl")
dst_data_curated = os.path.join(OUT_DIR, "sft_tool_calling_curated.jsonl")

shutil.copy2(src_data, dst_data_v2)
shutil.copy2(src_data, dst_data_curated)
print(f"1. Dataset SFT v2 copiado: {os.path.getsize(dst_data_v2) / 1024 / 1024:.2f} MB")

# 2. Copiar script de treino SFT
src_script = os.path.join(ROOT, "scripts", "train_sft_tool_qwen3loop.py")
dst_script = os.path.join(OUT_DIR, "train_sft_tool_qwen3loop.py")
shutil.copy2(src_script, dst_script)
print("2. Script de treino train_sft_tool_qwen3loop.py copiado.")

# 3. Copiar engine qwen3loop_python
src_engine = os.path.join(ROOT, "engine", "qwen3loop_python")
dst_engine = os.path.join(OUT_DIR, "qwen3loop_python")
shutil.copytree(src_engine, dst_engine)
print("3. Engine qwen3loop_python copiada.")

# 4. Copiar catálogo de ferramentas e bateria holdout v2
os.makedirs(os.path.join(OUT_DIR, "tool_catalog"), exist_ok=True)
shutil.copy2(os.path.join(ROOT, "data", "tool_catalog", "tools_schema.py"), os.path.join(OUT_DIR, "tool_catalog", "tools_schema.py"))
shutil.copy2(os.path.join(ROOT, "data", "eval_benchmark_battery_v2.jsonl"), os.path.join(OUT_DIR, "eval_benchmark_battery_v2.jsonl"))
print("4. Catálogo de ferramentas e Bateria Holdout v2 copiados.")

# 5. Metadados do dataset Kaggle
metadata = {
    "title": "qwen3loop-tool-sft-data",
    "id": "lordnyx0/qwen3loop-tool-sft-data",
    "licenses": [{"name": "CC0-1.0"}]
}
with open(os.path.join(OUT_DIR, "dataset-metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("\nBundle kaggle_sft_data preparado com sucesso!")
print("=" * 70)
