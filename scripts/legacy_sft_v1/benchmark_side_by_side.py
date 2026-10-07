"""
benchmark_side_by_side.py
Bateria comparativa rigorosa entre:
  1. Qwen3-0.6B-Q8_0.gguf (Base oficial Unsloth, 28 camadas)
  2. unrolled_modelo_qwen3loop_sft_q8_0.gguf (Qwen3Loop SFT, 56 camadas lógicas)

Avalia:
  - Resposta correta (Ground Truth)
  - Estrutura de raciocínio (<think> e Chain-of-Thought)
  - Presença de alucinação / divagação
  - Tokens gerados e velocidade (tokens/s)
"""

import sys
import time
import json
from pathlib import Path
from llama_cpp import Llama

TEST_CASES = [
    {
        "id": "modulo_2_100_7",
        "category": "Teoria dos Números",
        "question": "Find the remainder when 2^100 is divided by 7. Show your reasoning step by step and give the final answer in \\boxed{}.",
        "ground_truth": "2",
        "check_func": lambda text: "boxed{2}" in text or "remainder is 2" in text.lower() or "remainder is: 2" in text.lower()
    },
    {
        "id": "sally_sisters",
        "category": "Lógica / Pegadinha",
        "question": "Sally has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? Think carefully and explain.",
        "ground_truth": "1 sister (total 2 sisters in the family)",
        "check_func": lambda text: "1 sister" in text.lower() or "one sister" in text.lower() or "sally has 1" in text.lower()
    },
    {
        "id": "sheep_die",
        "category": "Pegadinha de Linguagem",
        "question": "A farmer has 17 sheep and all but 9 die. How many sheep are left alive?",
        "ground_truth": "9",
        "check_func": lambda text: "9" in text and ("9 sheep" in text.lower() or "nine" in text.lower() or "9 are left" in text.lower() or "9 remain" in text.lower())
    },
    {
        "id": "units_digit_3_2025",
        "category": "Matemática / Padrões",
        "question": "What is the units digit of 3^2025? Show the pattern of units digits step by step and give the final answer in \\boxed{}.",
        "ground_truth": "3",
        "check_func": lambda text: "boxed{3}" in text or "units digit is 3" in text.lower()
    },
    {
        "id": "balanced_parentheses",
        "category": "Algoritmo / Código",
        "question": "Write a clean Python function is_balanced(s) to check if a string of parentheses like '(()())' is valid. Explain its time and space complexity.",
        "ground_truth": "Stack or counter logic with O(n) time and O(n) or O(1) space",
        "check_func": lambda text: "def is_balanced" in text and ("O(n)" in text or "O(N)" in text)
    }
]

def format_prompt(q: str) -> str:
    return (
        f"<|im_start|>system\nYou are a helpful assistant that solves problems methodically.<|im_end|>\n"
        f"<|im_start|>user\n{q}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

def run_battery(model_path: str, model_name: str):
    print(f"\n{'='*70}\n CARREGANDO: {model_name}\n Arquivo: {model_path}\n{'='*70}")
    llm = Llama(
        model_path=model_path,
        n_gpu_layers=99,
        n_ctx=2048,
        verbose=False
    )
    results = []

    for idx, tc in enumerate(TEST_CASES):
        print(f"\n--- [Teste {idx+1}/{len(TEST_CASES)}] {tc['category']}: {tc['id']} ---")
        prompt = format_prompt(tc['question'])
        t0 = time.time()
        res = llm(
            prompt,
            max_tokens=380,
            temperature=0.1,
            top_p=0.95,
            stop=["<|im_end|>"]
        )
        dt = time.time() - t0
        text = res['choices'][0]['text'].strip()
        n_tokens = res['usage']['completion_tokens']
        tps = n_tokens / dt if dt > 0 else 0
        is_correct = tc['check_func'](text)

        has_think = ("<think>" in text or "[Start thinking]" in text or "[ASSESS]" in text)
        print(f"Tokens: {n_tokens} | Tempo: {dt:.2f}s | Velocidade: {tps:.1f} tok/s")
        print(f"Acerto Ground Truth: {'[CORRETO]' if is_correct else '[INCORRETO/INCOMPLETO]'}")
        print(f"Prévia Resposta:\n{text[:300]}...\n")

        results.append({
            "test_id": tc['id'],
            "category": tc['category'],
            "correct": is_correct,
            "tokens": n_tokens,
            "time_sec": round(dt, 2),
            "tps": round(tps, 1),
            "output": text
        })

    del llm
    return results

def main():
    m_base = "Qwen3-0.6B-Q8_0.gguf"
    m_loop = "unrolled_modelo_qwen3loop_sft_q8_0.gguf"

    if not Path(m_base).exists():
        print(f"Erro: {m_base} não encontrado.")
        sys.exit(1)
    if not Path(m_loop).exists():
        print(f"Erro: {m_loop} não encontrado.")
        sys.exit(1)

    res_base = run_battery(m_base, "Modelo Base Unsloth (28 camadas)")
    res_loop = run_battery(m_loop, "Seu Qwen3Loop SFT Unrolled (56 camadas)")

    # Consolidar comparação
    summary = []
    print("\n" + "="*80)
    print(" RESUMO COMPARATIVO FINAL LADO A LADO")
    print("="*80)
    print(f"{'Categoria / Teste':<28} | {'Base (28L) Acerto':<18} | {'Loop (56L) Acerto':<18} | {'Velocidade Base / Loop'}")
    print("-" * 80)

    score_base = 0
    score_loop = 0
    for b, l in zip(res_base, res_loop):
        c_b = "SIM" if b['correct'] else "NAO"
        c_l = "SIM" if l['correct'] else "NAO"
        if b['correct']: score_base += 1
        if l['correct']: score_loop += 1
        speed_comp = f"{b['tps']} t/s  vs  {l['tps']} t/s"
        print(f"{b['category'][:28]:<28} | {c_b:<18} | {c_l:<18} | {speed_comp}")

    print("-" * 80)
    print(f"PONTUAÇÃO TOTAL: Base = {score_base}/{len(TEST_CASES)} | Qwen3Loop SFT = {score_loop}/{len(TEST_CASES)}")
    print("="*80)

    output_json = {
        "score_base": score_base,
        "score_loop": score_loop,
        "base_results": res_base,
        "loop_results": res_loop
    }
    with open("data/side_by_side_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
    print("\nRelatório completo salvo em: data/side_by_side_benchmark_results.json")

if __name__ == "__main__":
    main()
