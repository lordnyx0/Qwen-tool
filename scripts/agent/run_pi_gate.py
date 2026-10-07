"""run_pi_gate.py — bateria Pi (5 tarefas, sem python): primeira bateria do modelo generic.

Uso: python scripts/agent/run_pi_gate.py [--model GGUF] [--temp 0.8] [--seed 42]
Cada tarefa roda em root fresco com fixtures; scoring por regras; sai runs/agent_gate_pi5_<ts>.jsonl.
"""
import argparse
import json
import os
import shutil
import sys
import time

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CUR, "..", "eval"))
sys.path.insert(0, os.path.join(CUR, "..", "pipeline"))
import agent
from agent import EXEC, SYSTEM, TOOLS
from benchmark_battery_v2 import extract_tool_calls_from_text

BASE_ROOT = r"C:\Users\Nyx\AppData\Local\Temp\opencode"
TASKS = json.load(open(os.path.join(CUR, "pi_tasks.json"), encoding="utf-8"))
ABSENT_WORDS = ("no such", "not found", "missing", "absent", "does not exist",
                "nao existe", "nao encontrado", "inexistente", "cannot find")


def run_task(llm, tok, task, temp, seed, max_turns=6):
    root = os.path.join(BASE_ROOT, "pi_root_" + task["id"])
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)
    for name, content in (task.get("fixtures") or {}).items():
        with open(os.path.join(root, name), "w", encoding="utf-8") as f:
            f.write(content)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": task["prompt"]}]
    turns, last_call, final = [], None, ""
    for _ in range(max_turns):
        pt = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                                      add_generation_prompt=True)
        out = llm(pt, max_tokens=384, temperature=temp, seed=seed,
                  repeat_penalty=1.1, stop=["<|im_end|>", "<|endoftext|>"])
        raw = out["choices"][0]["text"].strip()
        tcs, _, _ = extract_tool_calls_from_text(raw)
        if not tcs:
            final = raw
            break
        tc = tcs[0]
        nm, ag = tc.get("name"), tc.get("arguments", {})
        key = (nm, json.dumps(ag, sort_keys=True))
        if key == last_call:
            turns.append({"call": nm, "args": ag, "stop": "identical-retry"})
            break
        last_call = key
        fn = EXEC.get(nm)
        try:
            ok, code, so, se = fn(root, ag) if fn else (False, 127, "", f"unknown tool {nm}")
        except Exception as e:
            ok, code, so, se = False, 1, "", f"{type(e).__name__}: {e}"
        content = (so if ok else (se or so))[:1500]
        turns.append({"call": nm, "args": ag, "ok": ok})
        messages.append({"role": "assistant", "content": raw,
                         "tool_calls": [{"id": f"call_{len(turns):03d}", "type": "function",
                                         "function": {"name": nm, "arguments": json.dumps(ag)}}]})
        messages.append({"role": "tool", "name": nm,
                          "tool_call_id": f"call_{len(turns):03d}", "content": content})
    return {"turns": turns, "final": final, "root": root}


def score(task, res):
    exp, turns, final = task["expect"], res["turns"], res["final"]
    if exp.startswith("dir_exists:"):
        d = exp.split(":", 1)[1]
        ok = os.path.isdir(os.path.join(res["root"], d))
        return ok, "dir presente" if ok else "dir ausente"
    if exp == "graceful_absence":
        ok = bool(final) and any(w in final.lower() for w in ABSENT_WORDS)
        return ok, "ausencia reportada" if ok else "sem reporte de ausencia"
    if exp.startswith("final_contains:"):
        want = exp.split(":", 1)[1]
        ok = want in (final or "")
        return ok, f"contem {want}" if ok else "sem valor esperado"
    if exp == "direct_answer":
        mut = [t for t in turns if t.get("call") in ("bash", "powershell", "execute_python") and t.get("ok")]
        ok = bool(final) and not mut and not any(t.get("stop") for t in turns)
        return ok, "resposta direta" if ok else "chamou tool ou travou"
    if exp == "list_ok":
        ok = any(t.get("call") in ("list_dir", "list_directory") and t.get("ok") for t in turns)
        return ok, "listagem ok" if ok else "sem listagem"
    return False, "expect desconhecido"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=agent.DEFAULT_MODEL)
    ap.add_argument("--tok", default="checkpoints/qwen3loop_tool_sft_final")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from llama_cpp import Llama
    from transformers import AutoTokenizer
    llm = Llama(model_path=a.model, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    tok = AutoTokenizer.from_pretrained(a.tok, trust_remote_code=True)
    recs, score_sum = [], 0.0
    for task in TASKS:
        res = run_task(llm, tok, task, a.temp, a.seed)
        ok, note = score(task, res)
        score_sum += 1.0 if ok else 0.0
        recs.append({"id": task["id"], "prompt": task["prompt"], "pass": ok, "note": note,
                     "turns": res["turns"], "final": (res["final"] or "")[:500]})
        print(f"{task['id']}: {'PASS' if ok else 'FAIL'} ({note})", flush=True)
    print(f"PI-5: {score_sum:.1f}/{len(TASKS)}", flush=True)
    os.makedirs("runs", exist_ok=True)
    fp = f"runs/agent_gate_pi5_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    with open(fp, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("log:", fp, flush=True)


if __name__ == "__main__":
    main()
