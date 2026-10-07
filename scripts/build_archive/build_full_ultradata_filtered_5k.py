import os
import sys
import json
import time
from multiprocessing import Pool, cpu_count
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(".")
TOKENIZER_DIR = os.path.join(ROOT_DIR, "checkpoints", "qwen3loop_sft_final")
CACHE_DIR = r"C:\Users\Nyx\.cache\huggingface\hub\datasets--openbmb--UltraData-SFT-Agent-2609\snapshots\f684cc1a9f3e19f6f4929102cd9b06cc0b895b8a\data\Tool_Use"
OUTPUT_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_ALL_5K = os.path.join(OUTPUT_DIR, "sft_ultradata_tool_use_all_5k.jsonl")
OUTPUT_CLEAN_NON_SAC = os.path.join(OUTPUT_DIR, "sft_ultradata_tool_use_non_sac_5k.jsonl")

MAX_TOKENS = 5000

def process_shard(part_idx):
    filename = f"Tool_Use_part-{part_idx}-of-9.jsonl"
    filepath = os.path.join(CACHE_DIR, filename)
    if not os.path.exists(filepath):
        return {"part": part_idx, "error": f"File not found: {filepath}"}
    
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    
    approved_samples = []
    total = 0
    passed = 0
    too_long = 0
    sac_count = 0
    non_sac_count = 0
    errors = 0
    
    t0 = time.time()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            total += 1
            try:
                item = json.loads(line)
            except Exception:
                errors += 1
                continue
                
            msgs = item.get("messages", [])
            tools = item.get("tools", [])
            source = item.get("source", "unknown")
            uuid = item.get("uuid", f"part{part_idx}_{total}")
            
            # Checar se termina em assistant
            if not msgs or msgs[-1].get("role") != "assistant":
                continue
                
            # Pre-filter grosseiro por caracteres (se o texto total for gigantesco, já descarta)
            total_chars = sum(len(m.get("content") or "") for m in msgs) + sum(len(str(t)) for t in tools)
            if total_chars > 35000:
                too_long += 1
                continue
                
            # Normalizar tool_calls
            for m in msgs:
                if m.get("tool_calls"):
                    for tc in m["tool_calls"]:
                        fn = tc.get("function", {})
                        if isinstance(fn.get("arguments"), dict):
                            fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
                            
            # Renderizar e medir tokens com retorno estrito de lista
            try:
                tokens = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=True, return_dict=False)
                t_len = len(tokens)
                if t_len > MAX_TOKENS:
                    too_long += 1
                    continue
            except Exception:
                errors += 1
                continue
                
            # Identificar se é do template SAC com viés de verificação de identidade
            is_sac = False
            if source == "areal_tau2" or (msgs and "customer service agent" in str(msgs[0].get("content", "")).lower()):
                is_sac = True
                sac_count += 1
            else:
                non_sac_count += 1
                
            item["token_length"] = t_len
            item["is_sac_policy"] = is_sac
            approved_samples.append(item)
            passed += 1

    dt = time.time() - t0
    return {
        "part": part_idx,
        "total": total,
        "passed": passed,
        "too_long": too_long,
        "sac_count": sac_count,
        "non_sac_count": non_sac_count,
        "errors": errors,
        "elapsed": dt,
        "samples": approved_samples
    }

if __name__ == "__main__":
    print("=" * 70, flush=True)
    print(f"PROCESSAMENTO RIGOROSO DE TODOS OS 9 SHARDS DE Tool-Use (Max {MAX_TOKENS} tokens)", flush=True)
    print("=" * 70, flush=True)
    
    num_workers = min(6, cpu_count())
    print(f"Utilizando {num_workers} processos paralelos...", flush=True)
    
    t_start = time.time()
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_shard, range(1, 10))
        
    print("\n--- RESUMO DE PROCESSAMENTO POR SHARD ---", flush=True)
    grand_total = 0
    grand_passed = 0
    grand_too_long = 0
    grand_sac = 0
    grand_non_sac = 0
    
    all_approved = []
    non_sac_approved = []
    
    for r in sorted(results, key=lambda x: x["part"]):
        p = r["part"]
        tot = r.get("total", 0)
        pas = r.get("passed", 0)
        tl = r.get("too_long", 0)
        sac = r.get("sac_count", 0)
        ns = r.get("non_sac_count", 0)
        dt = r.get("elapsed", 0)
        
        grand_total += tot
        grand_passed += pas
        grand_too_long += tl
        grand_sac += sac
        grand_non_sac += ns
        
        for s in r.get("samples", []):
            all_approved.append(s)
            if not s.get("is_sac_policy", False):
                non_sac_approved.append(s)
                
        print(f"Shard {p}/9: {tot:,} lidos | {pas:,} aprovados ({pas/tot*100:.1f}%) | Non-SAC: {ns:,} | SAC: {sac:,} | >5k: {tl:,} | Tempo: {dt:.1f}s", flush=True)
        
    print("\n" + "=" * 70, flush=True)
    print(f"TOTAL GERAL AVALIADO: {grand_total:,} trajetórias", flush=True)
    print(f"APROVADOS (<= {MAX_TOKENS} tokens): {grand_passed:,} ({grand_passed/grand_total*100:.1f}%)", flush=True)
    print(f"  - Non-SAC (Bancos de dados, finanças, APIs, virtual tools): {grand_non_sac:,}", flush=True)
    print(f"  - SAC / Retail (Atendimento e políticas com verificação): {grand_sac:,}", flush=True)
    print(f"DESCARTADOS (> {MAX_TOKENS} tokens): {grand_too_long:,} ({grand_too_long/grand_total*100:.1f}%)", flush=True)
    print(f"Tempo total decorrido: {time.time() - t_start:.1f}s", flush=True)
    print("=" * 70, flush=True)
    
    # Salvar dataset geral filtrado
    print(f"\nSalvando todas as amostras <= {MAX_TOKENS} em: {OUTPUT_ALL_5K}...", flush=True)
    with open(OUTPUT_ALL_5K, "w", encoding="utf-8") as f:
        for s in all_approved:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"-> Salvo com sucesso! ({os.path.getsize(OUTPUT_ALL_5K)/1024/1024:.2f} MB, {len(all_approved):,} amostras)", flush=True)
    
    # Salvar dataset limpo sem o vício de SAC
    print(f"\nSalvando amostras Não-SAC (livres de vício) em: {OUTPUT_CLEAN_NON_SAC}...", flush=True)
    with open(OUTPUT_CLEAN_NON_SAC, "w", encoding="utf-8") as f:
        for s in non_sac_approved:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"-> Salvo com sucesso! ({os.path.getsize(OUTPUT_CLEAN_NON_SAC)/1024/1024:.2f} MB, {len(non_sac_approved):,} amostras)", flush=True)
