"""
test_openjev_scoring.py
Testa o AlexWortega/openjev como classificador NLI para anotar:
- difficulty
- needs_recurrence
- recurrence_value
- error_type
- quality
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
SUBFOLDER = "qwen3.5-0.8b-nli-v5"

print("Carregando OpenJev no dispositivo:", DEVICE)
tok = AutoTokenizer.from_pretrained("AlexWortega/openjev", subfolder=SUBFOLDER)
model = AutoModelForSequenceClassification.from_pretrained(
    "AlexWortega/openjev", subfolder=SUBFOLDER, dtype=torch.float16
).to(DEVICE)
model.eval()

# Pegar 1 exemplo do candidates_raw_25k
with open(r"data\candidates_raw_25k.jsonl", "r", encoding="utf-8") as f:
    sample = json.loads(f.readline())

problem = sample["prompt"]
answer = sample["response"]

print("\n--- EXEMPLO DE TESTE ---")
print("Problema:", problem[:150], "...")
print("Resposta:", answer[:150], "...")

# Hypotheses
hypotheses = {
    "multi_step": "Solving this problem requires multiple sequential steps of deduction or algorithmic planning.",
    "needs_second_pass": "A second pass or recurrent deliberation would significantly improve the solution or prevent subtle edge case errors.",
    "degenerate": "The text contains repetitive loops, verbatim repetition, or degenerate formatting.",
    "high_quality": "This is a correct, well-structured, and high-quality response.",
    "reasoning_error": "The response contains errors in logical or mathematical reasoning."
}

results = {}
with torch.inference_mode():
    for name, hyp in hypotheses.items():
        # Formato NLI Cross-Encoder: premise, hypothesis
        inputs = tok(
            f"Problem: {problem}\n\nSolution: {answer[:1200]}",
            hyp,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(DEVICE)
        
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        # id2label: {0: 'contradiction', 1: 'entailment', 2: 'neutral'}
        p_contra = probs[0].item()
        p_entail = probs[1].item()
        p_neutral = probs[2].item()
        
        results[name] = {
            "entailment": round(p_entail, 3),
            "contradiction": round(p_contra, 3),
            "neutral": round(p_neutral, 3)
        }

print("\n--- RESULTADOS NLI OPENJEV ---")
for k, v in results.items():
    print(f"  {k:20s}: entailment={v['entailment']:.3f}, contra={v['contradiction']:.3f}, neutral={v['neutral']:.3f}")

# Mapear para o schema do usuário
p_steps = results["multi_step"]["entailment"]
p_second = results["needs_second_pass"]["entailment"]
p_deg = results["degenerate"]["entailment"]
p_qual = results["high_quality"]["entailment"]

# Dificuldade
if p_steps >= 0.70:
    difficulty = "hard"
elif p_steps >= 0.35:
    difficulty = "medium"
else:
    difficulty = "easy"

# Recorrência
needs_recurrence = p_second >= 0.40
if p_second >= 0.75:
    recurrence_value = 3
elif p_second >= 0.40:
    recurrence_value = 2
else:
    recurrence_value = 1

# Error Type
error_type = "none"
if results["reasoning_error"]["entailment"] > 0.5:
    error_type = "reasoning"

final_meta = {
    "problem": problem,
    "answer": answer,
    "difficulty": difficulty,
    "needs_recurrence": needs_recurrence,
    "recurrence_value": recurrence_value,
    "error_type": error_type,
    "quality": p_qual
}

print("\n--- SCHEMA FINAL GERADO ---")
print(json.dumps({k: v if k not in ("problem", "answer") else v[:100] + "..." for k, v in final_meta.items()}, indent=2, ensure_ascii=False))
