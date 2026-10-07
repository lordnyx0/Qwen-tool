"""
curate_and_filter_10k.py
Aplica a curadoria fina sobre os candidatos brutos de candidates_raw_25k.jsonl:
1. Filtro de integridade estrutural (fechamento de blocos ```, tags <think> e [ASSESS])
2. Filtro de comprimento e densidade (evita amostras curtas demais ou truncadas)
3. Detector de repetição / degeneração (n-gram entropy filter)
4. Ranqueamento e balanceamento:
   - Top 5.500 agêntico (Eurus-2)
   - Top 5.500 coding CoT (Open-R1)
   Total: 11.000 amostras de alta fidelidade para o SFT do Qwen3Loop.
"""

import os
import json
import random
from collections import Counter

DATA_DIR = r"C:\Users\Nyx\Downloads\Diversos\qwen\data"
INPUT_FILE = os.path.join(DATA_DIR, "candidates_raw_25k.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "sft_curated_11k.jsonl")

def evaluate_quality(item):
    prompt = item.get("prompt", "")
    resp = item.get("response", "")
    source = item.get("source", "")
    
    score = 100.0
    
    # 1. Integridade de blocos de código
    code_blocks = resp.count("```")
    if code_blocks % 2 != 0:
        return -1.0  # Bloco de código quebrado/não fechado
        
    total_len = len(prompt) + len(resp)
    # Limite ideal para modelo 0.6B (evita estourar contexto e economiza VRAM)
    if total_len < 300 or total_len > 5500:
        return -1.0
        
    # 2. Estrutura de raciocínio por fonte
    if source in ("open-r1-code", "open-r1-codeforces"):
        if "<think>" in resp and "</think>" in resp:
            think_part = resp.split("</think>")[0].replace("<think>", "").strip()
            ans_part = resp.split("</think>")[1].strip()
            if len(think_part) < 100 or len(ans_part) < 50:
                return -1.0
            ratio = len(think_part) / max(1, len(resp))
            if 0.20 <= ratio <= 0.80:
                score += 25.0
            else:
                score -= 10.0
            if "```" in ans_part:
                score += 15.0
        elif "```" in resp:
            score += 10.0
        else:
            return -1.0
            
    elif source == "eurus-2-agentic":
        tags = [t for t in ["[ASSESS]", "[ADVANCE]", "[VERIFY]", "[EXECUTE]"] if t in resp]
        if "[ASSESS]" not in tags or len(tags) < 2:
            return -1.0
        score += len(tags) * 12.0  # Bonifica decomposição agêntica rica
        if "```" in resp:
            score += 10.0
            
    # 3. Penalidade de degeneração/repetição de texto
    words = resp.lower().split()
    if len(words) > 60:
        four_grams = [tuple(words[i:i+4]) for i in range(len(words)-3)]
        if four_grams:
            unique_ratio = len(set(four_grams)) / len(four_grams)
            if unique_ratio < 0.68:  # Repetição excessiva
                return -1.0
            score += unique_ratio * 20.0
            
    return score

def run_curation():
    if not os.path.exists(INPUT_FILE):
        print(f"Erro: Arquivo {INPUT_FILE} não encontrado.")
        return

    print("=" * 60)
    print(" INICIANDO FILTRAGEM E CURADORIA DAS 11.000 MELHORES AMOSTRAS")
    print("=" * 60)
    
    candidates_eurus = []
    candidates_openr1 = []
    
    total_read = 0
    discarded = 0
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            total_read += 1
            try:
                item = json.loads(line)
            except Exception:
                continue
                
            score = evaluate_quality(item)
            if score < 0:
                discarded += 1
                continue
                
            item["quality_score"] = score
            if item.get("source") == "eurus-2-agentic":
                candidates_eurus.append(item)
            else:
                candidates_openr1.append(item)
                
    print(f"Total de candidatos lidos: {total_read:,}")
    print(f"Descartados por qualidade/comprimento: {discarded:,}")
    print(f"Aprovados Eurus-2 (Agêntico): {len(candidates_eurus):,}")
    print(f"Aprovados Open-R1 (Coding CoT): {len(candidates_openr1):,}")
    
    # Ordenar por score decrescente
    candidates_eurus.sort(key=lambda x: x["quality_score"], reverse=True)
    candidates_openr1.sort(key=lambda x: x["quality_score"], reverse=True)
    
    # Seleção equilibrada
    target_each = 5500
    selected_eurus = candidates_eurus[:target_each]
    selected_openr1 = candidates_openr1[:target_each]
    
    final_mix = selected_eurus + selected_openr1
    random.seed(42)
    random.shuffle(final_mix)
    
    print(f"\nSalvando mix curado final ({len(final_mix):,} amostras) em:")
    print(f"-> {OUTPUT_FILE}")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for row in final_mix:
            # Salva no formato padrão ChatML limpo
            entry = {
                "id": row.get("id"),
                "source": row.get("source"),
                "task": row.get("task"),
                "quality_score": round(row.get("quality_score", 0), 2),
                "messages": row.get("messages")
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    size_mb = os.path.getsize(OUTPUT_FILE) / 1024**2
    print(f"\nSUCESSO! Arquivo pronto: {size_mb:.2f} MB")
    print(f"Composição:")
    print(f"  - Agêntico (Eurus-2): {len(selected_eurus):,} amostras")
    print(f"  - Coding CoT (Open-R1): {len(selected_openr1):,} amostras")

if __name__ == "__main__":
    run_curation()
