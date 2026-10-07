import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("checkpoints/qwen3loop_sft_final", use_fast=True)
data_path = "data/sft_ultradata_tool_use_8k.jsonl"

with open(data_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

total = len(lines)
lengths = []
active_tokens_8k = []
active_tokens_6k = []
zero_active_6k = 0
truncated_assistant_6k = 0

im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
assistant_id = tokenizer.convert_tokens_to_ids("assistant")
nl_id = tokenizer.convert_tokens_to_ids("\n")

def get_labels(full_ids):
    labels = [-100] * len(full_ids)
    i = 0
    while i < len(full_ids):
        if full_ids[i] == im_start_id and i + 1 < len(full_ids) and full_ids[i+1] == assistant_id:
            i += 2
            if i < len(full_ids) and (full_ids[i] == nl_id or full_ids[i] == 198):
                i += 1
            while i < len(full_ids):
                if full_ids[i] == im_end_id:
                    labels[i] = full_ids[i]
                    i += 1
                    break
                else:
                    labels[i] = full_ids[i]
                    i += 1
        else:
            i += 1
    return labels

# Sample first 300 to be fast (or all 3000 if fast)
count = min(500, total)
for idx in range(count):
    row = lines[idx]
    msgs = row.get("messages", [])
    tools = row.get("tools", None)
    try:
        full_txt = tokenizer.apply_chat_template(msgs, tools=tools, add_generation_prompt=False, tokenize=False)
        full_ids = tokenizer.encode(full_txt, add_special_tokens=False)
    except Exception:
        continue
    
    L = len(full_ids)
    lengths.append(L)
    
    # At 8192
    ids_8k = full_ids[:8192]
    lab_8k = get_labels(ids_8k)
    act_8k = sum(1 for x in lab_8k if x != -100)
    active_tokens_8k.append(act_8k)
    
    # At 6144
    ids_6k = full_ids[:6144]
    lab_6k = get_labels(ids_6k)
    act_6k = sum(1 for x in lab_6k if x != -100)
    active_tokens_6k.append(act_6k)
    
    if act_6k == 0:
        zero_active_6k += 1
    elif act_6k < act_8k:
        truncated_assistant_6k += 1

print(f"Total amostras analisadas: {len(lengths)} de {total}")
print(f"Comprimento Total (Tokens):")
print(f"  Min: {min(lengths)}")
print(f"  Max: {max(lengths)}")
print(f"  Média: {sum(lengths)/len(lengths):.1f}")
print(f"  <= 5500 tokens: {sum(1 for x in lengths if x <= 5500)} ({sum(1 for x in lengths if x <= 5500)/len(lengths)*100:.1f}%)")
print(f"  <= 6144 tokens: {sum(1 for x in lengths if x <= 6144)} ({sum(1 for x in lengths if x <= 6144)/len(lengths)*100:.1f}%)")
print(f"  <= 7000 tokens: {sum(1 for x in lengths if x <= 7000)} ({sum(1 for x in lengths if x <= 7000)/len(lengths)*100:.1f}%)")
print(f"  > 6144 tokens:  {sum(1 for x in lengths if x > 6144)} ({sum(1 for x in lengths if x > 6144)/len(lengths)*100:.1f}%)")
print(f"Impacto nos turnos do Assistente com max_len=6144:")
print(f"  Amostras que perdem 100% do turno do assistente (0 tokens ativos): {zero_active_6k} ({zero_active_6k/len(lengths)*100:.2f}%)")
print(f"  Amostras com assistente parcialmente truncado: {truncated_assistant_6k} ({truncated_assistant_6k/len(lengths)*100:.2f}%)")
print(f"  Amostras 100% intactas (assistente completo): {len(lengths) - zero_active_6k - truncated_assistant_6k} ({(len(lengths) - zero_active_6k - truncated_assistant_6k)/len(lengths)*100:.2f}%)")
