"""
append_codeforces_10k.py
Complementa a base bruta candidates_raw_25k.jsonl com 10.000 amostras
de Coding CoT de alta densidade do open-r1/CodeForces-CoTs.
"""

import os
import json
import time
from datasets import load_dataset

RAW_FILE = r"C:\Users\Nyx\Downloads\Diversos\qwen\data\candidates_raw_25k.jsonl"

def clean_text(t):
    if not t:
        return ""
    return str(t).strip()

def append_codeforces(target=10000):
    print("=" * 60)
    print(f" COLETANDO {target:,} AMOSTRAS DE CODING CoT (open-r1/CodeForces-CoTs)")
    print("=" * 60)
    t0 = time.time()
    
    ds = load_dataset("open-r1/CodeForces-CoTs", split="train", streaming=True)
    count = 0
    
    with open(RAW_FILE, "a", encoding="utf-8") as out_f:
        for row in ds:
            msgs = row.get("messages", [])
            if len(msgs) < 2:
                continue
                
            user_msg = clean_text(msgs[0].get("content", ""))
            gpt_msg = clean_text(msgs[1].get("content", ""))
            
            if not user_msg or not gpt_msg:
                continue
            # Limite para 0.6B (150 a 6000 chars)
            if len(gpt_msg) < 200 or len(gpt_msg) > 6000 or len(user_msg) > 2500:
                continue
                
            has_think = "<think>" in gpt_msg and "</think>" in gpt_msg
            has_code = "```" in gpt_msg
            if not (has_think and has_code):
                continue
                
            messages = [
                {"role": "system", "content": "You are an expert competitive programmer and autonomous coding assistant."},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": gpt_msg}
            ]
            
            item = {
                "id": f"codeforces_{count}",
                "source": "open-r1-codeforces",
                "task": "coding",
                "prompt": user_msg,
                "response": gpt_msg,
                "messages": messages
            }
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            
            if count % 1000 == 0:
                out_f.flush()
                print(f"  [CodeForces] {count}/{target} coletados ({time.time() - t0:.1f}s)", flush=True)
                
            if count >= target:
                break
                
        out_f.flush()
        
    print(f"\nConcluído! {count} amostras de CodeForces adicionadas.")
    print(f"Tamanho final do arquivo: {os.path.getsize(RAW_FILE)/1024**2:.2f} MB")

if __name__ == "__main__":
    append_codeforces()
