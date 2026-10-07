"""
test_laya_full_scoring.py
Testa a bateria completa de 8 perguntas no Laya sobre uma amostra real
"""

import json
import laya

print("Carregando Laya...")
agent = laya.load()

# Perguntas completas
LAYA_QUESTIONS = {
    "multi_step": {
        "type": "noul",
        "instructions": "Does solving this problem require multiple sequential reasoning or algorithmic deduction steps?"
    },
    "needs_second_pass": {
        "type": "noul",
        "instructions": "Would a second pass or recurrent deliberation significantly improve the answer or prevent subtle edge-case errors?"
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
    "degenerate": {
        "type": "noul",
        "instructions": "Does the text exhibit repetitive loops, verbatim repetition, or degenerate formatting?"
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
        "criteria": ["poor or incorrect", "acceptable", "good", "flawless or competition level"]
    }
}

# Pegar uma amostra do candidates_raw_25k
with open(r"data\candidates_raw_25k.jsonl", "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())

problem = sample["prompt"]
answer = sample["response"]

state = f"Problem:\n{problem}\n\nCandidate Solution:\n{answer[:2000]}"

print("Executando predict no Laya...")
res = agent.predict(state, LAYA_QUESTIONS)

ans = res.get("answers", {})
print("\n--- RAW LAYA ANSWERS ---")
print(json.dumps(ans, indent=2))


p_multi = ans.get("multi_step", {}).get("noul", 0.5)
if p_multi >= 0.70:
    difficulty = "hard"
elif p_multi >= 0.35:
    difficulty = "medium"
else:
    difficulty = "easy"

p_rec = ans.get("needs_second_pass", {}).get("noul", 0.5)
needs_recurrence = bool(p_rec >= 0.40)
if p_rec >= 0.75:
    recurrence_value = 3
elif p_rec >= 0.40:
    recurrence_value = 2
else:
    recurrence_value = 1

has_self_correction = bool(ans.get("has_self_correction", {}).get("noul", 0.0) >= 0.50)
is_verifiable = bool(ans.get("is_verifiable", {}).get("noul", 0.0) >= 0.50)
code_completeness = bool(ans.get("code_completeness", {}).get("noul", 0.0) >= 0.50)
is_degenerate = bool(ans.get("degenerate", {}).get("noul", 0.0) >= 0.50)

# Choice mapping
err_val = ans.get("error_type", {})
error_type = err_val.get("choice", "none")

quality = round(ans.get("quality", {}).get("score", 0.8), 3)

final_item = {
    "problem": problem[:120] + "...",
    "answer": answer[:120] + "...",
    "difficulty": difficulty,
    "needs_recurrence": needs_recurrence,
    "recurrence_value": recurrence_value,
    "has_self_correction": has_self_correction,
    "is_verifiable": is_verifiable,
    "code_completeness": code_completeness,
    "is_degenerate": is_degenerate,
    "error_type": error_type,
    "quality": quality
}

print("\n--- SCHEMA RESULTANTE DO LAYA ---")
print(json.dumps(final_item, indent=2, ensure_ascii=False))
