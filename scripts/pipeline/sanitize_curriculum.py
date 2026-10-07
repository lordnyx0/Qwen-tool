"""
sanitize_curriculum.py
Varre e corrige todas as amostras do dataset de currículo para garantir que
`tools` seja sempre uma lista válida de dicionários compatível com o Jinja template
do tokenizer do Qwen. Testa o chat template em 100% das amostras.
"""

import os
import json
from transformers import AutoTokenizer

CURRICULUM_DIR = os.path.abspath("data/rl_curriculum")
MODEL_DIR = os.path.abspath("checkpoints/qwen3loop_sft_final")

print("Carregando tokenizer de:", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

files = [
    "level_1_foundation.jsonl",
    "level_2_discrimination.jsonl",
    "level_3_reasoning_python.jsonl",
    "eval_benchmark_battery.jsonl"
]

def sanitize_tools(raw_tools):
    if raw_tools is None:
        return []
    if isinstance(raw_tools, str):
        raw_tools = raw_tools.strip()
        if not raw_tools:
            return []
        try:
            parsed = json.loads(raw_tools)
            raw_tools = parsed
        except Exception:
            return []
    
    if isinstance(raw_tools, dict):
        raw_tools = [raw_tools]
        
    if not isinstance(raw_tools, list):
        return []
        
    cleaned_tools = []
    for t in raw_tools:
        if isinstance(t, str):
            try:
                t = json.loads(t)
            except Exception:
                continue
        if not isinstance(t, dict):
            continue
            
        # Garantir estrutura {"type": "function", "function": {...}} se vier plano
        if "name" in t and "type" not in t:
            t = {
                "type": "function",
                "function": t
            }
        cleaned_tools.append(t)
        
    return cleaned_tools

for fname in files:
    fpath = os.path.join(CURRICULUM_DIR, fname)
    if not os.path.exists(fpath):
        print(f"Arquivo não encontrado: {fpath}")
        continue
        
    print(f"\nProcessando {fname}...")
    valid_samples = []
    fixed_count = 0
    discarded_count = 0
    
    with open(fpath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except Exception as e:
                print(f"  Erro de JSON linha {line_no}: {e}")
                discarded_count += 1
                continue
                
            orig_tools = sample.get("tools")
            cleaned_tools = sanitize_tools(orig_tools)
            if cleaned_tools != orig_tools:
                fixed_count += 1
            sample["tools"] = cleaned_tools
            
            # Testar renderização no tokenizer
            try:
                rendered = tokenizer.apply_chat_template(
                    sample["messages"],
                    tools=sample["tools"] if sample["tools"] else None,
                    add_generation_prompt=True,
                    tokenize=False
                )
                valid_samples.append(sample)
            except Exception as e:
                print(f"  Falha no chat template na linha {line_no}: {e}")
                discarded_count += 1
                
    # Sobrescrever com os dados limpos e validados
    with open(fpath, "w", encoding="utf-8") as f:
        for s in valid_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            
    print(f"  -> Concluído {fname}: {len(valid_samples)} amostras válidas mantidas ({fixed_count} corrigidas, {discarded_count} descartadas).")

print("\nTodas as baterias de currículo foram limpas e 100% testadas com o template do Qwen!")
