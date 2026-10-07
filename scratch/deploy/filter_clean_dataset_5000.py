import json
import os
import sys
import time

max_len = 5000
print(f"Iniciando filtragem estrita do dataset para max_len = {max_len} tokens (Zero truncamento)...", flush=True)

input_path = "data/sft_ultradata_tool_use_8k.jsonl"
output_path = "data/sft_ultradata_tool_use_clean_5k.jsonl"
kaggle_data_path = "kaggle_sft_data/sft_ultradata_tool_use_8k.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    raw_lines = [json.loads(line) for line in f if line.strip()]

total = len(raw_lines)
print(f"Total de amostras brutas lidas: {total}", flush=True)

clean_samples = []
discarded_too_long = 0
discarded_zero_active = 0
discarded_invalid = 0

for idx, row in enumerate(raw_lines):
    # token_length já foi rigorosamente medido e validado com o tokenizer Qwen
    t_len = row.get("token_length", 99999)
    if t_len > max_len:
        discarded_too_long += 1
        continue
    
    msgs = row.get("messages", [])
    if not msgs or msgs[-1].get("role") != "assistant":
        discarded_invalid += 1
        continue
        
    clean_samples.append(row)

print(f"\n--- RELATÓRIO FINAL DE FILTRAGEM (max_len = {max_len}) ---", flush=True)
print(f"Total original: {total}", flush=True)
print(f"Descartados por exceder {max_len} tokens: {discarded_too_long} ({discarded_too_long/total*100:.1f}%)", flush=True)
print(f"Descartados inválidos/incompletos: {discarded_invalid}", flush=True)
print(f"Aprovados 100% íntegros (zero truncamento): {len(clean_samples)} ({len(clean_samples)/total*100:.1f}%)", flush=True)

# Salvar versão limpa local
with open(output_path, "w", encoding="utf-8") as f:
    for s in clean_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# Substituir também no diretório kaggle_sft_data que será enviado ao Kaggle
with open(kaggle_data_path, "w", encoding="utf-8") as f:
    for s in clean_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\nArquivos salvos com sucesso em:\n - {output_path}\n - {kaggle_data_path}", flush=True)
