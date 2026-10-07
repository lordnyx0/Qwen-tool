"""
build_sft_mix.py
Constrói o dataset de SFT curado para o Qwen3Loop:
- Domínios: Agêntico (PRIME-RL/Eurus-2) + Coding CoT (open-r1/Mixture-of-Thoughts - code)
- Curadoria: Validação de estrutura de raciocínio (<think> / [ASSESS]), filtragem de tamanho (150 a 1536 tokens)
- Formato: ChatML padrão Qwen (<|im_start|>system/user/assistant)
"""

import os
import json
import re
from datasets import load_dataset

OUTPUT_DIR = r"C:\Users\Nyx\Downloads\Diversos\qwen\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "sft_agentic_code_mix.jsonl")

def clean_text(t):
    if not t:
        return ""
    return str(t).strip()

def process_eurus_agentic(max_samples=7500):
    print(f"\n[1/2] Baixando e curando PRIME-RL/Eurus-2-SFT-Data (Agêntico)...")
    ds = load_dataset("PRIME-RL/Eurus-2-SFT-Data", split="train", streaming=True)
    samples = []
    
    for row in ds:
        system = clean_text(row.get("system", ""))
        convs = row.get("conversations", [])
        if len(convs) < 2:
            continue
        
        # Validar primeiro turno
        user_msg = clean_text(convs[0].get("value", ""))
        gpt_msg = clean_text(convs[1].get("value", ""))
        
        # Curadoria de qualidade: precisa conter raciocínio agêntico estruturado
        if not user_msg or not gpt_msg:
            continue
        if len(gpt_msg) < 150 or len(gpt_msg) > 6000:
            continue
        
        # Verificar se tem blocos de ação agêntica ou CoT
        has_agentic = any(tag in gpt_msg for tag in ["[ASSESS]", "[ADVANCE]", "[VERIFY]", "[EXECUTE]", "```"])
        if not has_agentic:
            continue
            
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        else:
            messages.append({"role": "system", "content": "You are a helpful and precise reasoning agent."})
            
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": gpt_msg})
        
        samples.append({
            "source": "eurus-2-agentic",
            "task": row.get("task", "agentic"),
            "messages": messages
        })
        
        if len(samples) >= max_samples:
            break
            
    print(f"  -> Coletados {len(samples)} exemplos agênticos.")
    return samples

def process_openr1_code(max_samples=7500):
    print(f"\n[2/2] Baixando e curando open-r1/Mixture-of-Thoughts (Coding CoT)...")
    ds = load_dataset("open-r1/Mixture-of-Thoughts", "code", split="train", streaming=True)
    samples = []
    
    for row in ds:
        msgs = row.get("messages", [])
        if len(msgs) < 2:
            continue
            
        # Pegar user e assistant
        user_content = ""
        asst_content = ""
        for m in msgs:
            if m.get("role") == "user" and not user_content:
                user_content = clean_text(m.get("content"))
            elif m.get("role") == "assistant" and not asst_content:
                asst_content = clean_text(m.get("content"))
                
        if not user_content or not asst_content:
            continue
        if len(asst_content) < 200 or len(asst_content) > 6000:
            continue
            
        # Curadoria de Coding: deve ter <think> ou código python/estruturado
        has_code_or_think = "<think>" in asst_content or "```" in asst_content
        if not has_code_or_think:
            continue
            
        messages = [
            {"role": "system", "content": "You are an expert autonomous software engineer and coding assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": asst_content}
        ]
        
        samples.append({
            "source": "open-r1-code",
            "task": "coding",
            "messages": messages
        })
        
        if len(samples) >= max_samples:
            break
            
    print(f"  -> Coletados {len(samples)} exemplos de Coding CoT.")
    return samples

def main():
    eurus = process_eurus_agentic(max_samples=1000) # Teste inicial rápido de 1k + 1k
    openr1 = process_openr1_code(max_samples=1000)
    
    mix = eurus + openr1
    import random
    random.seed(42)
    random.shuffle(mix)
    
    print(f"\nSalvando {len(mix)} amostras em {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in mix:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    print(f"OK! Arquivo gerado com sucesso: {os.path.getsize(OUTPUT_FILE) / 1024**2:.2f} MB")

if __name__ == "__main__":
    main()
