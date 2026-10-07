import json
import os
import sys
import time
from transformers import AutoTokenizer

max_len = 4608
print(f"Iniciando filtragem estrita do dataset para max_len = {max_len} tokens (Zero truncamento)...", flush=True)

tokenizer = AutoTokenizer.from_pretrained("checkpoints/qwen3loop_sft_final", use_fast=True)
input_path = "data/sft_ultradata_tool_use_8k.jsonl"
output_path = "data/sft_ultradata_tool_use_clean_4608.jsonl"
kaggle_data_path = "kaggle_sft_data/sft_ultradata_tool_use_8k.jsonl"

im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
assistant_id = tokenizer.convert_tokens_to_ids("assistant")
nl_id = tokenizer.convert_tokens_to_ids("\n")

def count_active_tokens(full_ids):
    count = 0
    i = 0
    while i < len(full_ids):
        if full_ids[i] == im_start_id and i + 1 < len(full_ids) and full_ids[i+1] == assistant_id:
            i += 2
            if i < len(full_ids) and (full_ids[i] == nl_id or full_ids[i] == 198):
                i += 1
            while i < len(full_ids):
                if full_ids[i] == im_end_id:
                    count += 1
                    i += 1
                    break
                else:
                    count += 1
                    i += 1
        else:
            i += 1
    return count

with open(input_path, "r", encoding="utf-8") as f:
    raw_lines = [json.loads(line) for line in f if line.strip()]

total = len(raw_lines)
print(f"Total de amostras brutas lidas: {total}", flush=True)

clean_samples = []
discarded_too_long = 0
discarded_zero_active = 0
discarded_error = 0

t0 = time.time()
for idx, row in enumerate(raw_lines):
    msgs = row.get("messages", [])
    tools = row.get("tools") or None
    try:
        full_txt = tokenizer.apply_chat_template(msgs, tools=tools, add_generation_prompt=False, tokenize=False)
        full_ids = tokenizer.encode(full_txt, add_special_tokens=False)
    except Exception as e:
        discarded_error += 1
        continue
    
    # Regra estrita: se exceder max_len, descarta para evitar corte no meio do assistente
    if len(full_ids) > max_len:
        discarded_too_long += 1
        continue
        
    act_tokens = count_active_tokens(full_ids)
    if act_tokens == 0:
        discarded_zero_active += 1
        continue
        
    clean_samples.append(row)

    if (idx + 1) % 500 == 0 or (idx + 1) == total:
        print(f"Processado: {idx + 1}/{total} | Aprovados: {len(clean_samples)} | Descartados >{max_len}: {discarded_too_long} | Tempo: {time.time()-t0:.1f}s", flush=True)

print(f"\n--- RELATÓRIO FINAL DE FILTRAGEM (max_len = {max_len}) ---", flush=True)
print(f"Total original: {total}", flush=True)
print(f"Descartados por exceder {max_len} tokens: {discarded_too_long} ({discarded_too_long/total*100:.1f}%)", flush=True)
print(f"Descartados com 0 tokens de assistente: {discarded_zero_active}", flush=True)
print(f"Descartados por erro de template: {discarded_error}", flush=True)
print(f"Aprovados 100% íntegros (com <|im_end|>): {len(clean_samples)} ({len(clean_samples)/total*100:.1f}%)", flush=True)

# Salvar versão limpa local
with open(output_path, "w", encoding="utf-8") as f:
    for s in clean_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# Substituir também no diretório kaggle_sft_data que será enviado ao Kaggle
with open(kaggle_data_path, "w", encoding="utf-8") as f:
    for s in clean_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Salvo em:\n - {output_path}\n - {kaggle_data_path}\nConcluído com sucesso!", flush=True)
