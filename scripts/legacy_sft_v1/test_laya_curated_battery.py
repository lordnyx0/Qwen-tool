"""
test_laya_curated_battery.py
Testa a bateria aprimorada de perguntas no Laya sobre 3 amostras reais de origens distintas:
1. Eurus-2 (Agêntico)
2. Open-R1 (Coding CoT)
3. CodeForces (Algorítmico)
"""

import json
import laya

print("Carregando Laya...")
agent = laya.load()

# Bateria aprimorada de meta-supervisão
ADVANCED_QUESTIONS = {
    # 1. Complexidade e profundidade
    "multi_step": {
        "type": "noul",
        "instructions": "Does solving this problem require multiple sequential reasoning or algorithmic deduction steps?"
    },
    # 2. Early-Exit vs Recorrência
    "early_exit_safe": {
        "type": "noul",
        "instructions": "Can this problem be reliably answered in a single shallow pass without deep iterative deliberation?"
    },
    "needs_second_pass": {
        "type": "noul",
        "instructions": "Would a second pass or recurrent deliberation significantly improve the answer or prevent subtle edge-case errors?"
    },
    # 3. Estratégia de Raciocínio
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
    # 4. Auto-correção e verificação
    "has_self_correction": {
        "type": "noul",
        "instructions": "Does the response explicitly contain mental verification, backtracking, hypothesis testing, or self-correction?"
    },
    "is_verifiable": {
        "type": "noul",
        "instructions": "Does this task have a single deterministic, verifiable ground-truth (e.g. math or code) rather than open-ended text?"
    },
    # 5. Código e Integridade
    "code_completeness": {
        "type": "noul",
        "instructions": "Is the provided code complete, functional, and free of lazy placeholders like '# TODO' or '... '?"
    },
    # 6. Diagnóstico de Erro
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
    # 7. Qualidade Global
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

# Coletar 3 amostras diferentes
samples = {}
with open(r"data\candidates_raw_25k.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        src = item.get("source", "unknown")
        if src not in samples:
            samples[src] = item
        if len(samples) >= 3:
            break

print(f"Testando em {len(samples)} tipos de fontes: {list(samples.keys())}")

for src, sample in samples.items():
    print("=" * 70)
    print(f"FONTE: {src}")
    problem = sample["prompt"]
    answer = sample["response"]
    
    # Montar estado limpo para o Laya (sem cortar abruptamente no meio de tokens)
    state = f"Problem:\n{problem[:1000]}\n\nCandidate Solution:\n{answer[:1500]}"
    
    res = agent.predict(state, ADVANCED_QUESTIONS)
    ans = res.get("answers", {})
    
    p_multi = ans.get("multi_step", {}).get("noul", 0.5)
    p_early = ans.get("early_exit_safe", {}).get("noul", 0.5)
    p_rec = ans.get("needs_second_pass", {}).get("noul", 0.5)
    
    # Determinar dynamic recurrence value (1, 2 ou 3 loops)
    if p_early >= 0.65 or p_rec < 0.35:
        recurrence_value = 1
    elif p_rec >= 0.70 or p_multi >= 0.85:
        recurrence_value = 3
    else:
        recurrence_value = 2
        
    difficulty = "hard" if p_multi >= 0.70 else ("medium" if p_multi >= 0.35 else "easy")
    
    summary = {
        "source": src,
        "difficulty": difficulty,
        "recurrence_value": recurrence_value,
        "needs_recurrence": recurrence_value > 1,
        "early_exit_safe": bool(p_early >= 0.50),
        "reasoning_strategy": ans.get("reasoning_strategy", {}).get("choice"),
        "has_self_correction": bool(ans.get("has_self_correction", {}).get("noul", 0) >= 0.50),
        "is_verifiable": bool(ans.get("is_verifiable", {}).get("noul", 0) >= 0.50),
        "code_completeness": bool(ans.get("code_completeness", {}).get("noul", 0) >= 0.50),
        "error_type": ans.get("error_type", {}).get("choice"),
        "quality_score": round(ans.get("quality", {}).get("score", 0.0), 3),
        "raw_probabilities": {
            "p_multi": p_multi,
            "p_early_exit": p_early,
            "p_second_pass": p_rec
        }
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

print("=" * 70)
print("Bateria concluída com sucesso!")
