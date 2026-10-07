import json
import os
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("checkpoints/qwen3loop_sft_final", use_fast=True)
input_path = "data/sft_ultradata_tool_use_8k.jsonl"
output_path = "data/sft_ultradata_tool_use_clean_6k.jsonl"
kaggle_data_path = "kaggle_sft_data/sft_ultradata_tool_use_8k.jsonl"

max_len = 6144

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
clean_samples = []
discarded = 0

for row in raw_lines:
    msgs = row.get("messages", [])
    tools = row.get("tools") or None
    try:
        full_txt = tokenizer.apply_chat_template(msgs, tools=tools, add_generation_prompt=False, tokenize=False)
        full_ids = tokenizer.encode(full_txt, add_special_tokens=False)
    except Exception:
        discarded += 1
        continue
    
    # Regra estrita: se exceder max_len, descarta para evitar corte no meio do assistente
    if len(full_ids) > max_len:
        discarded += 1
        continue
        
    act_tokens = count_active_tokens(full_ids)
    if act_tokens == 0:
        discarded += 1
        continue
        
    clean_samples.append(row)

print(f"Total original: {total}")
print(f"Descartados (> {max_len} tokens ou invalidos): {discarded}")
print(f"Aprovados 100% intactos: {len(clean_samples)} ({len(clean_samples)/total*100:.1f}%)")

# Salvar versão limpa local
with open(output_path, "w", encoding="utf-8") as f:
    for s in clean_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

# Substituir também no diretório kaggle_sft_data que será enviado ao Kaggle
with open(kaggle_data_path, "w", encoding="utf-8") as f:
    for s in clean_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Salvo em {output_path} e {kaggle_data_path} com sucesso.")
