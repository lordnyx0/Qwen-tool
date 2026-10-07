"""
audit_50_samples_per_level.py
Amostragem aleatória e verificação profunda de 50 exemplos de cada nível (N1, N2, N3)
do dataset sft_tool_calling_curated.jsonl.
"""

import json
import random
import re
import sys
import io
import contextlib
from transformers import AutoTokenizer

random.seed(1337)

dataset_path = "data/sft_tool_calling_curated.jsonl"
benchmark_path = "data/rl_curriculum/eval_benchmark_battery.jsonl"

with open(dataset_path, "r", encoding="utf-8") as f:
    all_data = [json.loads(line) for line in f]

with open(benchmark_path, "r", encoding="utf-8") as f:
    bench_data = [json.loads(line) for line in f]

bench_prompts = set()
for b in bench_data:
    for m in b.get("messages", []):
        if m["role"] == "user":
            bench_prompts.add(m["content"].strip().lower())

# Separar por nível
l1 = [d for d in all_data if d.get("difficulty") == 1]
l2 = [d for d in all_data if d.get("difficulty") == 2]
l3 = [d for d in all_data if d.get("difficulty") == 3]

print(f"Total no dataset: {len(all_data)} (L1: {len(l1)}, L2: {len(l2)}, L3: {len(l3)})")

s50_l1 = random.sample(l1, 50)
s50_l2 = random.sample(l2, 50)
s50_l3 = random.sample(l3, 50)

tok = AutoTokenizer.from_pretrained("model_src")

report = {
    "l1": {"total": 50, "valid_template": 0, "no_leak": 0, "abstentions": 0, "single_calls": 0, "valid_args": 0, "issues": []},
    "l2": {"total": 50, "valid_template": 0, "no_leak": 0, "multi_tools_present": 0, "valid_calls": 0, "valid_args": 0, "issues": []},
    "l3": {"total": 50, "valid_template": 0, "no_leak": 0, "valid_code_compile": 0, "valid_code_exec": 0, "valid_args": 0, "issues": []},
}

# -------------------------------------------------------------
# AUDITORIA NÍVEL 1
# -------------------------------------------------------------
for idx, s in enumerate(s50_l1):
    u = [m["content"] for m in s["messages"] if m["role"] == "user"][0]
    a = [m for m in s["messages"] if m["role"] == "assistant"][0]
    
    # 1. Leak check
    if u.strip().lower() not in bench_prompts:
        report["l1"]["no_leak"] += 1
    else:
        report["l1"]["issues"].append(f"Sample L1 #{idx} leaked benchmark prompt: {u}")

    # 2. Template check
    try:
        tok.apply_chat_template(s["messages"], tools=s.get("tools"), tokenize=False)
        report["l1"]["valid_template"] += 1
    except Exception as e:
        report["l1"]["issues"].append(f"Sample L1 #{idx} template error: {e}")

    # 3. Call vs Abstain check
    tcs = a.get("tool_calls", [])
    if not tcs:
        report["l1"]["abstentions"] += 1
        if not a.get("content"):
            report["l1"]["issues"].append(f"Sample L1 #{idx} abstention has empty content")
    else:
        report["l1"]["single_calls"] += 1
        fn = tcs[0].get("function", {})
        args = fn.get("arguments", "")
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
            if isinstance(parsed, dict) and len(parsed) > 0:
                report["l1"]["valid_args"] += 1
            else:
                report["l1"]["issues"].append(f"Sample L1 #{idx} empty args in tool call")
        except Exception as e:
            report["l1"]["issues"].append(f"Sample L1 #{idx} invalid JSON args: {e}")

# -------------------------------------------------------------
# AUDITORIA NÍVEL 2
# -------------------------------------------------------------
for idx, s in enumerate(s50_l2):
    u = [m["content"] for m in s["messages"] if m["role"] == "user"][0]
    a = [m for m in s["messages"] if m["role"] == "assistant"][0]
    
    # 1. Leak check
    if u.strip().lower() not in bench_prompts:
        report["l2"]["no_leak"] += 1
    else:
        report["l2"]["issues"].append(f"Sample L2 #{idx} leaked benchmark prompt: {u}")

    # 2. Template check
    try:
        tok.apply_chat_template(s["messages"], tools=s.get("tools"), tokenize=False)
        report["l2"]["valid_template"] += 1
    except Exception as e:
        report["l2"]["issues"].append(f"Sample L2 #{idx} template error: {e}")

    # 3. Tool discrimination & Schema check
    tools = s.get("tools", [])
    if len(tools) >= 2:
        report["l2"]["multi_tools_present"] += 1
    
    tcs = a.get("tool_calls", [])
    if tcs:
        report["l2"]["valid_calls"] += 1
        fn = tcs[0].get("function", {})
        args = fn.get("arguments", "")
        try:
            parsed = json.loads(args) if isinstance(args, str) else args
            if isinstance(parsed, dict):
                report["l2"]["valid_args"] += 1
            else:
                report["l2"]["issues"].append(f"Sample L2 #{idx} args is not a dict")
        except Exception as e:
            report["l2"]["issues"].append(f"Sample L2 #{idx} invalid JSON args: {e}")
    else:
        report["l2"]["issues"].append(f"Sample L2 #{idx} has no tool_calls")

# -------------------------------------------------------------
# AUDITORIA NÍVEL 3
# -------------------------------------------------------------
for idx, s in enumerate(s50_l3):
    u = [m["content"] for m in s["messages"] if m["role"] == "user"][0]
    a = [m for m in s["messages"] if m["role"] == "assistant"][0]
    
    # 1. Leak check
    if u.strip().lower() not in bench_prompts:
        report["l3"]["no_leak"] += 1
    else:
        report["l3"]["issues"].append(f"Sample L3 #{idx} leaked benchmark prompt: {u}")

    # 2. Template check
    try:
        tok.apply_chat_template(s["messages"], tools=s.get("tools"), tokenize=False)
        report["l3"]["valid_template"] += 1
    except Exception as e:
        report["l3"]["issues"].append(f"Sample L3 #{idx} template error: {e}")

    # 3. Code extraction, compilation, and execution
    tcs = a.get("tool_calls", [])
    if tcs:
        fn = tcs[0].get("function", {})
        if fn.get("name") == "execute_python":
            args = fn.get("arguments", "")
            try:
                parsed = json.loads(args) if isinstance(args, str) else args
                code = parsed.get("code", "")
                report["l3"]["valid_args"] += 1
                
                # Compilar
                compile(code, "<string>", "exec")
                report["l3"]["valid_code_compile"] += 1
                
                # Executar com captura de stdout
                f = io.StringIO()
                with contextlib.redirect_stdout(f):
                    loc = {}
                    exec(code, {}, loc)
                out_str = f.getvalue().strip()
                if out_str or len(loc) > 0:
                    report["l3"]["valid_code_exec"] += 1
                else:
                    report["l3"]["issues"].append(f"Sample L3 #{idx} execution yielded empty stdout/vars")
            except Exception as e:
                report["l3"]["issues"].append(f"Sample L3 #{idx} code error: {e}")
        else:
            report["l3"]["issues"].append(f"Sample L3 #{idx} tool is not execute_python: {fn.get('name')}")
    else:
        report["l3"]["issues"].append(f"Sample L3 #{idx} has no tool_calls")

# Salvar relatório detalhado
with open("logs/audit_150_samples_report.json", "w", encoding="utf-8") as f:
    json.dump({
        "summary": report,
        "sample_preview": {
            "l1": s50_l1[:5],
            "l2": s50_l2[:5],
            "l3": s50_l3[:5]
        }
    }, f, indent=2, ensure_ascii=False)

print("\n=======================================================")
print("RESULTADOS DA AUDITORIA (50 AMOSTRAS POR NÍVEL)")
print("=======================================================")
print(f"NÍVEL 1 (50 amostras sorteadas):")
print(f"  - Zero Vazamento Holdout:     {report['l1']['no_leak']}/50")
print(f"  - Template Tokenizer Válido:  {report['l1']['valid_template']}/50")
print(f"  - Abstenções Corretas:        {report['l1']['abstentions']}")
print(f"  - Single-Tools Corretas:      {report['l1']['single_calls']}")
print(f"  - Argumentos Válidos:         {report['l1']['valid_args']}/{report['l1']['single_calls']}")
print(f"  - Falhas/Inconsistências:     {len(report['l1']['issues'])}")

print(f"\nNÍVEL 2 (50 amostras sorteadas):")
print(f"  - Zero Vazamento Holdout:     {report['l2']['no_leak']}/50")
print(f"  - Template Tokenizer Válido:  {report['l2']['valid_template']}/50")
print(f"  - Múltiplas Ferramentas:      {report['l2']['multi_tools_present']}/50")
print(f"  - Tool Calls Executadas:      {report['l2']['valid_calls']}/50")
print(f"  - Argumentos Válidos:         {report['l2']['valid_args']}/50")
print(f"  - Falhas/Inconsistências:     {len(report['l2']['issues'])}")

print(f"\nNÍVEL 3 (50 amostras sorteadas):")
print(f"  - Zero Vazamento Holdout:     {report['l3']['no_leak']}/50")
print(f"  - Template Tokenizer Válido:  {report['l3']['valid_template']}/50")
print(f"  - Códigos Compiláveis:        {report['l3']['valid_code_compile']}/50")
print(f"  - Códigos Executáveis (Real): {report['l3']['valid_code_exec']}/50")
print(f"  - Argumentos Válidos:         {report['l3']['valid_args']}/50")
print(f"  - Falhas/Inconsistências:     {len(report['l3']['issues'])}")

print("\n--- Exemplos do Nível 1 ---")
for i, s in enumerate(s50_l1[:3]):
    u = [m["content"] for m in s["messages"] if m["role"] == "user"][0]
    a = [m for m in s["messages"] if m["role"] == "assistant"][0]
    tc = a.get("tool_calls")
    ans = f"CALL {tc[0]['function']['name']}({tc[0]['function']['arguments']})" if tc else f"ABSTAIN: {a.get('content')[:60]}..."
    print(f"  [{i+1}] User: {u[:70]}...\n      Asst: {ans}")

print("\n--- Exemplos do Nível 2 ---")
for i, s in enumerate(s50_l2[:3]):
    u = [m["content"] for m in s["messages"] if m["role"] == "user"][0]
    a = [m for m in s["messages"] if m["role"] == "assistant"][0]
    tc = a.get("tool_calls", [{}])[0].get("function", {})
    tools_available = [t["function"]["name"] for t in s.get("tools", [])]
    print(f"  [{i+1}] Tools Disponíveis ({len(tools_available)}): {tools_available[:4]}...")
    print(f"      User: {u[:70]}...\n      Asst: Chamou '{tc.get('name')}' com args: {tc.get('arguments')[:60]}...")

print("\n--- Exemplos do Nível 3 ---")
for i, s in enumerate(s50_l3[:3]):
    u = [m["content"] for m in s["messages"] if m["role"] == "user"][0]
    a = [m for m in s["messages"] if m["role"] == "assistant"][0]
    tc = a.get("tool_calls", [{}])[0].get("function", {})
    code = json.loads(tc.get("arguments", "{}")).get("code", "")
    code_first_line = code.split("\n")[0]
    print(f"  [{i+1}] User: {u[:70]}...\n      Python Code: {code_first_line} ...")
