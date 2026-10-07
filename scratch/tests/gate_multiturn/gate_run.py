"""Scratch: gate multi-turn real (v38 Q8 + harness_local no sandbox).
6 prompts x temps {0.0, 0.8}, seed 42, max 6 turnos. Registra calls, repeticoes,
respostas finais e efeitos reais. Sem rede, sem repo real.
"""
import copy
import json
import os
import shutil
import sys

sys.path.insert(0, "scripts/eval")
sys.path.insert(0, "scripts/pipeline")
from benchmark_battery_v2 import extract_tool_calls_from_text
from harness_local import run_read_file, run_list_dir, run_powershell
from llama_cpp import Llama
from transformers import AutoTokenizer

SB = "C:/Users/Nyx/AppData/Local/Temp/opencode/gate_sandbox"
MODEL = os.environ.get("GATE_GGUF",
                        "models/gguf/unrolled_modelo_qwen3loop_tool_sft_v38_ep2_q8_0.gguf")
TOK = "kaggle_kernel_output/sft_v33/qwen3loop_tool_curriculum_3000_final/epoch-2"

SYS = ("You are a helpful assistant with access to tools. Use them when needed "
       "to complete the user's request. Be concise and direct.")
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Reads the content of a single file. Never use for directories.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "File path"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Execute bash shell commands in Linux terminal and view stdout/stderr output.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string", "description": "Command line"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "powershell",
        "description": "Runs legitimate Windows administration commands and returns output.",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string", "description": "Command"}},
                       "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "Lists the structured contents of a directory.",
        "parameters": {"type": "object",
                       "properties": {"path": {"type": "string", "description": "Directory path"}},
                       "required": ["path"]}}},
]
RUN = {"read_file": run_read_file, "bash": None, "powershell": run_powershell,
       "list_dir": run_list_dir}

PROMPTS = [
    ("asfkap", "create a folder called asfkap"),
    ("enoent", "read arquivo_que_nao_existe.txt"),
    ("porta", "read serve_gate.bat and tell me which port the server uses"),
    ("pycalc", "run this python code and tell me the exact output: print(13**7)"),
    ("hi", "hi! how are you today?"),
    ("list", "list all files in this folder"),
]


def exec_call(name, args):
    if name == "bash":
        cmd = args.get("command", "")
        if cmd.startswith("mkdir "):
            d = cmd[6:].strip().strip("'\"")
            try:
                os.makedirs(os.path.join(SB, d), exist_ok=False)
                return True, 0, f"created directory '{d}'", ""
            except FileExistsError:
                return False, 1, "", f"mkdir: '{d}' exists"
            except Exception as e:
                return False, 1, "", f"{type(e).__name__}: {e}"
        if cmd.startswith("ls"):
            import subprocess
            tgt = cmd[2:].strip().strip("'\"") or "."
            try:
                r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                                    "-Command", f"Get-ChildItem '{tgt}' | Select-Object -ExpandProperty Name"],
                                   cwd=SB, capture_output=True, timeout=20)
                return r.returncode == 0, r.returncode, r.stdout.decode("utf-8", errors="replace"), ""
            except Exception as e:
                return False, 1, "", f"{type(e).__name__}: {e}"
        return False, 126, "", "REFUSED (harness demo: only mkdir/ls)"
    fn = RUN.get(name)
    if fn is None:
        return False, 127, "", f"unknown tool {name}"
    try:
        return fn(SB, args)
    except Exception as e:
        return False, 1, "", f"{type(e).__name__}: {e}"


def run_gate(temp):
    print(f"===== TEMP {temp} =====", flush=True)
    llm = Llama(model_path=MODEL, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    report = []
    for pid, prompt in PROMPTS:
        if pid == "asfkap":
            shutil.rmtree(os.path.join(SB, "asfkap"), ignore_errors=True)
        messages = [{"role": "system", "content": SYS},
                    {"role": "user", "content": prompt}]
        calls_log = []
        final = ""
        for turn in range(6):
            pt = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                                         add_generation_prompt=True)
            out = llm(pt, max_tokens=384, temperature=temp, seed=42,
                      repeat_penalty=1.0, stop=["<|im_end|>", "<|endoftext|>"])
            raw = out["choices"][0]["text"].strip()
            tcs, has_xml, errs = extract_tool_calls_from_text(raw)
            if not tcs:
                final = raw
                break
            for tc in tcs:
                nm = tc.get("name")
                ag = tc.get("arguments", {})
                ok, code, so, se = exec_call(nm, ag)
                calls_log.append(f"{nm} {json.dumps(ag, ensure_ascii=False)[:90]} -> {'OK' if ok else 'ERR:' + (se or so)[:80]}")
                messages.append({"role": "assistant", "content": raw,
                                 "tool_calls": [{"id": f"call_{len(calls_log)}", "type": "function",
                                                 "function": {"name": nm, "arguments": json.dumps(ag)}}]})
                messages.append({"role": "tool", "name": nm,
                                 "tool_call_id": f"call_{len(calls_log)}",
                                 "content": (so if ok else (se or so))[:1500]})
                raw = ""
                break
        rep = sum(1 for a, b in zip(calls_log, calls_log[1:])
                  if a.split(" -> ")[0] == b.split(" -> ")[0])
        print(f"--- {pid} t={temp}: {len(calls_log)} calls, {rep} repeats-identicos", flush=True)
        for c in calls_log[:8]:
            print(f"    {c}", flush=True)
        print(f"    FINAL: {final[:200].replace(chr(10), ' | ')}", flush=True)
        report.append({"id": pid, "temp": temp, "calls": calls_log, "final": final})
    with open(f"C:/Users/Nyx/AppData/Local/Temp/opencode/gate_{os.environ.get('GATE_TAG', 'v38')}_t{temp}.json",
              "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(f"GATE t={temp} FIM", flush=True)


if __name__ == "__main__":
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    run_gate(t)
