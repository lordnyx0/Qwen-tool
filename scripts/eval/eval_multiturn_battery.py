"""
eval_multiturn_battery.py

Bateria multi-turn de correcao apos observation (prefixo fixo deterministico):
user -> assistant(tool_call com defeito) -> tool(erro) -> modelo deve corrigir.
Escora a SEGUNDA chamada com a mesma logica do harness single-turn
(`benchmark_battery_v2.evaluate_sample`).

Prompt renderizado UMA vez (tokenizer com template tools) e reutilizado
byte-identico em todos os modelos -> comparacao controlada.
Modelos GGUF via llama-cpp-python com GPU (n_gpu_layers=-1).

Uso:
    python scripts/eval/eval_multiturn_battery.py \
        --model_path models/gguf/legacy_v1/unrolled_modelo_qwen3loop_tool_sft_q8_0.gguf models/gguf/Qwen3-0.6B-Q8_0.gguf \
        --model_names tool_sft_v1 qwen3_base
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark_battery_v2 as bb
from llama_cpp import Llama
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_BATTERY = os.path.join(ROOT, "data", "eval_multiturn_v1.jsonl")
DEFAULT_TOKENIZER = os.path.join(ROOT, "models", "gguf", "legacy_v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", nargs="+", required=True)
    ap.add_argument("--model_names", nargs="+", required=True)
    ap.add_argument("--battery_path", default=DEFAULT_BATTERY)
    ap.add_argument("--tokenizer_dir", default=DEFAULT_TOKENIZER)
    ap.add_argument("--output_file", default="data/eval_results/multiturn/multiturn_v1.json")
    ap.add_argument("--n_ctx", type=int, default=8192)
    ap.add_argument("--max_tokens", type=int, default=384)
    args = ap.parse_args()
    assert len(args.model_path) == len(args.model_names)

    cases = [json.loads(l) for l in open(args.battery_path, encoding="utf-8") if l.strip()]
    print(f"itens multi-turn: {len(cases)}", flush=True)

    print("Renderizando prompts (uma vez, identicos p/ todos os modelos)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    prompts = []
    for case in cases:
        prompts.append(tokenizer.apply_chat_template(
            case["messages"], tools=case.get("tools", []),
            tokenize=False, add_generation_prompt=True))

    all_results = {}
    for mp, name in zip(args.model_path, args.model_names):
        print("=" * 70, flush=True)
        print(f"Modelo: {name} ({mp})", flush=True)
        t0 = time.time()
        llm = Llama(model_path=mp, n_ctx=args.n_ctx, n_gpu_layers=-1, verbose=False)
        print("carregado (GPU).", flush=True)
        results = []
        cats = defaultdict(lambda: {"total": 0, "pass": 0, "format": 0, "select": 0})
        for idx, (case, prompt) in enumerate(zip(cases, prompts), 1):
            out = llm(prompt, max_tokens=args.max_tokens, temperature=0.0,
                      stop=["<|im_end|>", "<|endoftext|>"])
            raw = out["choices"][0]["text"].strip()
            ev = bb.evaluate_sample(case, raw)
            results.append(ev)
            c = case.get("category", "?")
            cats[c]["total"] += 1
            for k, f in (("pass", "overall_pass"), ("format", "format_ok"), ("select", "selection_ok")):
                if ev[f]:
                    cats[c][k] += 1
            mark = "[PASS]" if ev["overall_pass"] else "[FAIL]"
            print(f"[{idx:02d}/{len(cases):02d}] {mark} {case['id']} ({c}) -> {ev['diag'][:100]}", flush=True)
        dt = time.time() - t0
        tot = len(results)
        ps = sum(1 for r in results if r["overall_pass"])
        print(f"{name}: {ps}/{tot} ({ps/tot*100:.1f}%) em {dt:.0f}s", flush=True)
        all_results[name] = {"model": mp, "passed": ps, "total": tot,
                             "elapsed_s": dt, "categories": dict(cats), "details": results}
        del llm

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=1, ensure_ascii=False)
    print(f"salvo: {args.output_file}", flush=True)


if __name__ == "__main__":
    main()
