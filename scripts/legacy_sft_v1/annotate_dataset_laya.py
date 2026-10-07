"""
annotate_dataset_laya.py
Executa a anotação em lote das amostras curadas usando o Laya (System 1 non-autoregressive decision model).
Gera metadados completos de supervisão adaptativa:
- difficulty (easy, medium, hard)
- recurrence_value (1, 2, 3)
- needs_recurrence (bool)
- early_exit_safe (bool)
- reasoning_strategy (direct, analytical, algorithmic, agentic)
- has_self_correction (bool)
- is_verifiable (bool)
- code_completeness (bool)
- error_type (none, reasoning, knowledge, syntax)
- quality_score (0.0 a 3.0)
"""

import os
import sys
import json
import time
import torch
import laya

DATA_DIR = r"C:\Users\Nyx\Downloads\Diversos\qwen\data"
INPUT_FILE = os.path.join(DATA_DIR, "sft_curated_11k.jsonl")
OUTPUT_FILE = os.path.join(DATA_DIR, "sft_annotated_curated.jsonl")

LAYA_QUESTIONS = {
    "multi_step": {
        "type": "noul",
        "instructions": "Does solving this problem require multiple sequential reasoning or algorithmic deduction steps?"
    },
    "early_exit_safe": {
        "type": "noul",
        "instructions": "Can this problem be reliably answered in a single shallow pass without deep iterative deliberation?"
    },
    "needs_second_pass": {
        "type": "noul",
        "instructions": "Would a second pass or recurrent deliberation significantly improve the answer or prevent subtle edge-case errors?"
    },
    "reasoning_strategy": {
        "type": "choice",
        "instructions": "What primary problem-solving strategy does this response employ?",
        "criteria": {
            "direct": "direct factual explanation or concise answer",
            "analytical": "step-by-step mathematical deduction or proof",
            "algorithmic": "code implementation with data structure / logic analysis",
            "agentic": "multi-phase planning with assessment, verification, and action execution"
        }
    },
    "has_self_correction": {
        "type": "noul",
        "instructions": "Does the response explicitly contain mental verification, backtracking, hypothesis testing, or self-correction?"
    },
    "is_verifiable": {
        "type": "noul",
        "instructions": "Does this task have a single deterministic, verifiable ground-truth (e.g. math or code) rather than open-ended text?"
    },
    "code_completeness": {
        "type": "noul",
        "instructions": "Is the provided code complete, functional, and free of lazy placeholders like '# TODO' or '... '?"
    },
    "error_type": {
        "type": "choice",
        "instructions": "What kind of flaw or error is present in the solution, if any?",
        "criteria": {
            "none": "no error, the solution is correct",
            "reasoning": "logical, algorithmic, or mathematical reasoning error",
            "knowledge": "factual or API hallucination gap",
            "syntax": "syntax or formatting error"
        }
    },
    "quality": {
        "type": "score",
        "instructions": "Rate the overall technical depth, clarity, and correctness of this response.",
        "criteria": [
            "poor or incorrect",
            "acceptable",
            "good",
            "flawless or competition level"
        ]
    }
}

def annotate():
    if not os.path.exists(INPUT_FILE):
        print(f"Erro: Arquivo {INPUT_FILE} não encontrado.")
        return

    # Suporte a resumo incremental
    already_annotated = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for _ in f:
                already_annotated += 1
        print(f"Detectado progresso anterior: {already_annotated:,} amostras já anotadas.")

    print("Carregando Laya na GPU...")
    t_load = time.time()
    agent = laya.load()
    print(f"Laya pronto em {time.time() - t_load:.2f}s (Dispositivo: {agent.device})")

    # Contagem de linhas total
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        total_items = sum(1 for _ in f)
    print(f"Total de itens para processar: {total_items:,} (restam {total_items - already_annotated:,})")

    if already_annotated >= total_items:
        print("Dataset já totalmente anotado!")
        return

    # Contadores estatísticos
    stats = {
        "rec_1": 0, "rec_2": 0, "rec_3": 0,
        "diff_easy": 0, "diff_medium": 0, "diff_hard": 0,
        "self_correction": 0, "verifiable": 0, "complete_code": 0,
        "strategies": {}, "errors": {}
    }

    t0 = time.time()
    processed_this_run = 0

    mode = "a" if already_annotated > 0 else "w"
    with open(INPUT_FILE, "r", encoding="utf-8") as in_f, open(OUTPUT_FILE, mode, encoding="utf-8") as out_f:
        for idx, line in enumerate(in_f):
            if idx < already_annotated:
                continue

            try:
                row = json.loads(line)
            except Exception:
                continue

            # Extrair prompt e response dos messages
            msgs = row.get("messages", [])
            user_msg = ""
            asst_msg = ""
            for m in msgs:
                if m.get("role") == "user" and not user_msg:
                    user_msg = m.get("content", "")
                elif m.get("role") == "assistant" and not asst_msg:
                    asst_msg = m.get("content", "")

            # Fallback para campos diretos se existirem
            if not user_msg:
                user_msg = row.get("prompt", "")
            if not asst_msg:
                asst_msg = row.get("response", "")

            # Formatar estado de inferência para o Laya (máx ~1500 chars para velocidade ideal)
            state = f"Problem:\n{user_msg[:800]}\n\nCandidate Solution:\n{asst_msg[:1200]}"

            try:
                res = agent.predict(state, LAYA_QUESTIONS)
                ans = res.get("answers", {})
            except Exception as e:
                # Fallback seguro caso estoure contexto pontual
                ans = {}

            p_multi = ans.get("multi_step", {}).get("noul", 0.5)
            p_early = ans.get("early_exit_safe", {}).get("noul", 0.5)
            p_rec = ans.get("needs_second_pass", {}).get("noul", 0.5)

            # Índice contínuo de necessidade de recorrência: R = 0.50*rec + 0.40*multi - 0.30*early
            r_index = (0.50 * p_rec) + (0.40 * p_multi) - (0.30 * p_early)
            if r_index < 0.20:
                rec_val = 1
                stats["rec_1"] += 1
            elif r_index < 0.55:
                rec_val = 2
                stats["rec_2"] += 1
            else:
                rec_val = 3
                stats["rec_3"] += 1

            if p_multi >= 0.75:
                diff = "hard"
                stats["diff_hard"] += 1
            elif p_multi >= 0.35:
                diff = "medium"
                stats["diff_medium"] += 1
            else:
                diff = "easy"
                stats["diff_easy"] += 1

            strat = ans.get("reasoning_strategy", {}).get("choice", "algorithmic")
            stats["strategies"][strat] = stats["strategies"].get(strat, 0) + 1

            has_sc = bool(ans.get("has_self_correction", {}).get("noul", 0.0) >= 0.50)
            if has_sc:
                stats["self_correction"] += 1

            is_ver = bool(ans.get("is_verifiable", {}).get("noul", 0.0) >= 0.50)
            if is_ver:
                stats["verifiable"] += 1

            code_comp = bool(ans.get("code_completeness", {}).get("noul", 0.0) >= 0.50)
            if code_comp:
                stats["complete_code"] += 1

            err_t = ans.get("error_type", {}).get("choice", "none")
            stats["errors"][err_t] = stats["errors"].get(err_t, 0) + 1

            qual_score = round(float(ans.get("quality", {}).get("score", 1.8)), 3)

            meta = {
                "difficulty": diff,
                "needs_recurrence": rec_val > 1,
                "recurrence_value": rec_val,
                "early_exit_safe": bool(p_early >= 0.50),
                "reasoning_strategy": strat,
                "has_self_correction": has_sc,
                "is_verifiable": is_ver,
                "code_completeness": code_comp,
                "error_type": err_t,
                "quality_score": qual_score,
                "recurrence_index": round(float(r_index), 4),
                "raw_scores": {
                    "p_multi": round(float(p_multi), 4),
                    "p_early": round(float(p_early), 4),
                    "p_second_pass": round(float(p_rec), 4)
                }
            }

            row["meta"] = meta
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            processed_this_run += 1

            # Log de progresso periódico
            if processed_this_run % 250 == 0:
                out_f.flush()
                elapsed = time.time() - t0
                speed = processed_this_run / max(1e-3, elapsed)
                remaining = (total_items - already_annotated - processed_this_run) / max(1e-3, speed)
                current_total = already_annotated + processed_this_run
                print(f"[{current_total:,}/{total_items:,}] ({current_total/total_items*100:.1f}%) | "
                      f"Vel: {speed:.1f} amostras/s | Restante: {remaining/60:.1f} min | "
                      f"Loops: [1x: {stats['rec_1']}, 2x: {stats['rec_2']}, 3x: {stats['rec_3']}]", flush=True)

        out_f.flush()

    total_time = time.time() - t0
    final_count = already_annotated + processed_this_run
    print("\n" + "=" * 60)
    print(" ANOTAÇÃO META-SUPERVISIONADA CONCLUÍDA!")
    print(f" Amostras anotadas nesta sessão: {processed_this_run:,}")
    print(f" Total no arquivo final: {final_count:,}")
    print(f" Tempo total: {total_time:.1f}s ({total_time/60:.2f} min)")
    print(f" Arquivo salvo em: {OUTPUT_FILE}")
    print("\nDISTRIBUIÇÃO DE RECORRÊNCIA:")
    print(f"  - 1 Loop (Early Exit Seguro): {stats['rec_1']:,} ({stats['rec_1']/max(1, processed_this_run)*100:.1f}%)")
    print(f"  - 2 Loops (Recorrência Típica): {stats['rec_2']:,} ({stats['rec_2']/max(1, processed_this_run)*100:.1f}%)")
    print(f"  - 3 Loops (Raciocínio Profundo): {stats['rec_3']:,} ({stats['rec_3']/max(1, processed_this_run)*100:.1f}%)")
    print("\nDISTRIBUIÇÃO DE DIFICULDADE:")
    print(f"  - Easy: {stats['diff_easy']:,} | Medium: {stats['diff_medium']:,} | Hard: {stats['diff_hard']:,}")
    print(f"\nESTRATÉGIAS DE RESOLUÇÃO: {stats['strategies']}")
    print(f"ERROS DETECTADOS: {stats['errors']}")
    print("=" * 60)

if __name__ == "__main__":
    annotate()
