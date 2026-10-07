"""
harvest_candidates_25k.py
Coleta 25.000 candidatos brutos:
- 12.000 amostras de PRIME-RL/Eurus-2-SFT-Data (Agêntico)
- 13.000 amostras de open-r1/Mixture-of-Thoughts (Coding CoT)
Salva incrementalmente em data/candidates_raw_25k.jsonl
"""

import os
import json
import time
from datasets import load_dataset

DATA_DIR = r"C:\Users\Nyx\Downloads\Diversos\qwen\data"
os.makedirs(DATA_DIR, exist_ok=True)
RAW_FILE = os.path.join(DATA_DIR, "candidates_raw_25k.jsonl")

def clean_text(t):
    if not t:
        return ""
    return str(t).strip()

def harvest():
    print("=" * 60)
    print(" INICIANDO COLETA DE 25.000 CANDIDATOS BRUTOS")
    print("=" * 60)
    t0 = time.time()
    
    total_written = 0
    with open(RAW_FILE, "w", encoding="utf-8") as out_f:
        
        # 1. PRIME-RL/Eurus-2-SFT-Data (Alvo: 12.000)
        target_eurus = 12000
        count_eurus = 0
        print(f"\n[1/2] Coletando {target_eurus} candidatos agênticos (PRIME-RL/Eurus-2-SFT-Data)...")
        ds_eurus = load_dataset("PRIME-RL/Eurus-2-SFT-Data", split="train", streaming=True)
        
        for row in ds_eurus:
            system = clean_text(row.get("system", ""))
            convs = row.get("conversations", [])
            if len(convs) < 2:
                continue
                
            user_msg = clean_text(convs[0].get("value", ""))
            gpt_msg = clean_text(convs[1].get("value", ""))
            
            if not user_msg or not gpt_msg:
                continue
            # Filtro de tamanho para 0.6B (150 a 6000 caracteres)
            if len(gpt_msg) < 150 or len(gpt_msg) > 6000 or len(user_msg) > 2500:
                continue
                
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
            
            item = {
                "id": f"eurus_{count_eurus}",
                "source": "eurus-2-agentic",
                "task": row.get("task", "agentic"),
                "prompt": user_msg,
                "response": gpt_msg,
                "messages": messages
            }
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count_eurus += 1
            total_written += 1
            
            if count_eurus % 1000 == 0:
                out_f.flush()
                print(f"  [Eurus-2] {count_eurus}/{target_eurus} coletados ({time.time() - t0:.1f}s)", flush=True)
                
            if count_eurus >= target_eurus:
                break
                
        out_f.flush()
        print(f"  -> Concluído Eurus-2: {count_eurus} amostras.")
        
        # 2. open-r1/Mixture-of-Thoughts - code (Alvo: 13.000)
        target_openr1 = 13000
        count_openr1 = 0
        print(f"\n[2/2] Coletando {target_openr1} candidatos Coding CoT (open-r1/Mixture-of-Thoughts)...")
        ds_r1 = load_dataset("open-r1/Mixture-of-Thoughts", "code", split="train", streaming=True)
        
        for row in ds_r1:
            msgs = row.get("messages", [])
            if len(msgs) < 2:
                continue
                
            user_content = ""
            asst_content = ""
            for m in msgs:
                if m.get("role") == "user" and not user_content:
                    user_content = clean_text(m.get("content"))
                elif m.get("role") == "assistant" and not asst_content:
                    asst_content = clean_text(m.get("content"))
                    
            if not user_content or not asst_content:
                continue
            if len(asst_content) < 200 or len(asst_content) > 6000 or len(user_content) > 2500:
                continue
                
            has_code_or_think = "<think>" in asst_content or "```" in asst_content
            if not has_code_or_think:
                continue
                
            messages = [
                {"role": "system", "content": "You are an expert autonomous software engineer and coding assistant."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": asst_content}
            ]
            
            item = {
                "id": f"openr1_{count_openr1}",
                "source": "open-r1-code",
                "task": "coding",
                "prompt": user_content,
                "response": asst_content,
                "messages": messages
            }
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count_openr1 += 1
            total_written += 1
            
            if count_openr1 % 1000 == 0:
                out_f.flush()
                print(f"  [Open-R1] {count_openr1}/{target_openr1} coletados ({time.time() - t0:.1f}s)", flush=True)
                
            if count_openr1 >= target_openr1:
                break
                
        out_f.flush()
        print(f"  -> Concluído Open-R1: {count_openr1} amostras.")

    size_mb = os.path.getsize(RAW_FILE) / 1024**2
    t_tot = time.time() - t0
    print("\n" + "=" * 60)
    print(f" COLETA FINALIZADA COM SUCESSO!")
    print(f" Total de amostras salvas: {total_written:,}")
    print(f" Arquivo: {RAW_FILE} ({size_mb:.2f} MB)")
    print(f" Tempo decorrido: {t_tot:.1f}s ({t_tot/60:.2f} min)")
    print("=" * 60)

if __name__ == "__main__":
    harvest()
