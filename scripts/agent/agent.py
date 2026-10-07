"""Agent Hermes proprio: tools do treino, execucao real, disciplina de loop.
Uso: python scripts/agent/agent.py run "prompt" [--root DIR] [--temp 0.8]
       [--seed N] [--max-turns 6] [--model GGUF]
Catalogo = vocabulario real do treino (5 tools, descricoes modais).
"""
import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time

CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CUR, "..", "eval"))
sys.path.insert(0, os.path.join(CUR, "..", "pipeline"))
from benchmark_battery_v2 import extract_tool_calls_from_text
from harness_local import run_read_file, run_list_dir, run_powershell
from harness_local import run_python as run_execute_python
from harness_local import (run_read_text_file, run_read_media_file,
                           run_read_multiple_files, run_list_directory_with_sizes,
                           run_directory_tree, run_search_files, run_get_file_info,
                           run_list_allowed_directories, run_create_directory,
                           run_edit_file, run_move_file)

GITBASH_USR = "C:\\Program Files\\Git\\usr\\bin\\bash.exe"
GITBASH_BIN = "C:\\Program Files\\Git\\bin\\bash.exe"
GITBASH = GITBASH_USR if os.path.exists(GITBASH_USR) else GITBASH_BIN
BASH_BLOCK = re.compile(
    r"(rm\s+-[a-z]*r|: *\( *\) *\{|mkfs|dd\s+.*of=|curl.+|\|\s*(ba)?sh\b|wget.+|\|\s*sh\b|"
    r"\bdu\s+.*(/|[a-z]:)|\bfind\s+/)",
    re.I,
)

SYSTEM = ("You are an AI programming and workspace automation assistant. "
          "Use tools via structured tool_calls format. Be concise, accurate, and direct.")
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Reads the content of a single file. Never use for directories.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "Lists a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "bash",
        "description": "Execute bash shell commands in Linux terminal and view stdout/stderr output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Command line"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "powershell",
        "description": "Runs legitimate Windows administration commands (Get-ChildItem, Get-Service) and returns output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Command"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "execute_python",
        "description": "Runs Python code and returns stdout.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Python code"}}, "required": ["code"]}}},
]
DEFAULT_MODEL = "models/gguf/unrolled_modelo_qwen3loop_tool_sft_v38_ep2_q8_0.gguf"


def run_bash(root, args):
    cmd = args.get("command", "")
    if BASH_BLOCK.search(cmd):
        return False, 126, "", "REFUSED by agent blocklist"
    exe = GITBASH if os.path.exists(GITBASH) else "bash"
    # Git Bash herda PATH vazio do serviço: injeta coreutils (usr/bin) + sistema.
    env = dict(os.environ)
    git_root = os.path.dirname(os.path.dirname(exe))
    env["PATH"] = (os.path.join(git_root, "usr", "bin") + os.pathsep
                   + os.path.join(git_root, "bin") + os.pathsep
                   + env.get("PATH", ""))
    try:
        r = subprocess.run([exe, "-c", cmd], cwd=root, stdin=subprocess.DEVNULL,
                           capture_output=True, timeout=20, env=env)
        return r.returncode == 0, r.returncode, r.stdout.decode("utf-8", errors="replace"), r.stderr.decode("utf-8", errors="replace")
    except Exception as e:
        return False, 1, "", f"{type(e).__name__}: {e}"


EXEC = {"read_file": run_read_file, "list_dir": run_list_dir,
        "read_text_file": run_read_text_file,
        "read_media_file": run_read_media_file,
        "read_multiple_files": run_read_multiple_files,
        "list_directory": run_list_dir,
        "list_directory_with_sizes": run_list_directory_with_sizes,
        "directory_tree": run_directory_tree,
        "search_files": run_search_files,
        "get_file_info": run_get_file_info,
        "list_allowed_directories": run_list_allowed_directories,
        "create_directory": run_create_directory,
        "edit_file": run_edit_file,
        "move_file": run_move_file,
        "powershell": run_powershell, "execute_python": run_execute_python,
        "bash": run_bash}


def main():
    ap = argparse.ArgumentParser(prog="agent")
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("prompt")
    ap.add_argument("--root", default="C:/Users/Nyx/AppData/Local/Temp/opencode/agent_root")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-turns", type=int, default=6)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tok", default="checkpoints/qwen3loop_tool_sft_final")
    a = ap.parse_args()

    from llama_cpp import Llama
    from transformers import AutoTokenizer
    os.makedirs(a.root, exist_ok=True)
    llm = Llama(model_path=a.model, n_ctx=8192, n_gpu_layers=-1, verbose=False)
    tok = AutoTokenizer.from_pretrained(a.tok, trust_remote_code=True)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": a.prompt}]
    turns = []
    last_call = None
    final = ""
    for _ in range(a.max_turns):
        pt = tok.apply_chat_template(messages, tools=TOOLS, tokenize=False,
                                     add_generation_prompt=True)
        out = llm(pt, max_tokens=384, temperature=a.temp, seed=a.seed,
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
            turns.append({"stop": "identical-retry", "call": key[0]})
            break
        last_call = key
        fn = EXEC.get(nm)
        try:
            ok, code, so, se = fn(a.root, ag) if fn else (False, 127, "", f"unknown tool {nm}")
        except Exception as e:
            ok, code, so, se = False, 1, "", f"{type(e).__name__}: {e}"
        content = (so if ok else (se or so))[:1500]
        turns.append({"call": nm, "args": ag, "ok": ok, "out": content[:200]})
        messages.append({"role": "assistant", "content": raw,
                         "tool_calls": [{"id": f"call_{len(turns):03d}", "type": "function",
                                         "function": {"name": nm, "arguments": json.dumps(ag)}}]})
        messages.append({"role": "tool", "name": nm,
                         "tool_call_id": f"call_{len(turns):03d}", "content": content})
    rec = {"ts": time.strftime("%Y%m%d_%H%M%S"), "model": a.model, "temp": a.temp,
           "seed": a.seed, "prompt": a.prompt, "turns": turns, "final": final}
    os.makedirs("runs", exist_ok=True)
    fp = f"runs/agent_{rec['ts']}.jsonl"
    with open(fp, "w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"turns={len(turns)} final: {final[:250]}", flush=True)
    print("log:", fp, flush=True)


if __name__ == "__main__":
    main()
