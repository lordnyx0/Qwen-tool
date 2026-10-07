import json
import re
from collections import Counter

dataset_path = "data/sft_tool_calling_curated.jsonl"
benchmark_path = "data/rl_curriculum/eval_benchmark_battery.jsonl"

with open(dataset_path, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

with open(benchmark_path, "r", encoding="utf-8") as f:
    bench_samples = [json.loads(line) for line in f]

bench_prompts = set()
for b in bench_samples:
    for m in b.get("messages", []):
        if m["role"] == "user":
            bench_prompts.add(m["content"].strip().lower())

print(f"=== AUDITORIA DO DATASET SFT ===")
print(f"Total de amostras: {len(samples)}")
print(f"Total de itens no Benchmark Holdout: {len(bench_samples)}")

# 1. Checagem de Leakage (Vazamento de dados)
leakage = []
for idx, s in enumerate(samples):
    for m in s.get("messages", []):
        if m["role"] == "user":
            c = m["content"].strip().lower()
            if c in bench_prompts:
                leakage.append((idx, c))

print(f"\n1. VAZAMENTO DO BENCHMARK:")
print(f"   Amostras vazadas do holdout: {len(leakage)}")

# 2. Distribuição por Nível / Dificuldade
diff_counter = Counter()
tool_names = Counter()
call_vs_abstain = Counter()

for s in samples:
    diff = s.get("difficulty", 1)
    diff_counter[diff] += 1
    
    msgs = s.get("messages", [])
    has_tool_call = False
    for m in msgs:
        if m.get("role") == "assistant":
            tcs = m.get("tool_calls", [])
            if tcs:
                has_tool_call = True
                for tc in tcs:
                    fn = tc.get("function", {}).get("name", "unknown")
                    tool_names[fn] += 1
    
    if has_tool_call:
        call_vs_abstain["Tool Call Realizada"] += 1
    else:
        call_vs_abstain["Abstenção (Conversa/Resposta Direta)"] += 1

print(f"\n2. DISTRIBUIÇÃO DE DIFICULDADE (Dificuldade da Sonda):")
for k, v in sorted(diff_counter.items()):
    print(f"   Dificuldade {k}: {v} amostras ({v/len(samples)*100:.1f}%)")

print(f"\n3. TOOL CALL vs. ABSTENÇÃO NEGATIVA:")
for k, v in call_vs_abstain.items():
    print(f"   {k}: {v} amostras ({v/len(samples)*100:.1f}%)")

print(f"\n4. TOP FERRAMENTAS UTILIZADAS:")
for k, v in tool_names.most_common(12):
    print(f"   - {k}: {v} chamadas")

# 5. Integridade Estrutural e Sintática
invalid_json_args = 0
total_calls = 0
python_math_calls = 0
python_syntax_errors = 0

for idx, s in enumerate(samples):
    msgs = s.get("messages", [])
    for m in msgs:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls", []):
                total_calls += 1
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "")
                
                try:
                    parsed_args = json.loads(args) if isinstance(args, str) else args
                except Exception:
                    invalid_json_args += 1
                    parsed_args = {}
                
                if name == "execute_python":
                    python_math_calls += 1
                    code = parsed_args.get("code", "") if isinstance(parsed_args, dict) else ""
                    try:
                        compile(code, "<string>", "exec")
                    except Exception as e:
                        python_syntax_errors += 1

print(f"\n5. VALIDAÇÃO SINTÁTICA E EXECUÇÃO:")
print(f"   Total de chamadas de ferramenta: {total_calls}")
print(f"   Argumentos JSON inválidos: {invalid_json_args}")
print(f"   Chamadas a 'execute_python' (N3): {python_math_calls}")
print(f"   Erros de sintaxe Python no código gerado: {python_syntax_errors}")

# 6. Amostras representativas de cada nível
print(f"\n6. EXEMPLOS REPRESENTATIVOS:")
for target_diff in [1, 2, 3]:
    for s in samples:
        if s.get("difficulty") == target_diff:
            u_list = [m["content"] for m in s["messages"] if m["role"] == "user"]
            a_list = [m for m in s["messages"] if m["role"] == "assistant"]
            if u_list and a_list:
                u = u_list[0]
                a = a_list[0]
                tcs = a.get("tool_calls", [])
                tc_str = f"Tool: {tcs[0]['function']['name']}({tcs[0]['function']['arguments'][:40]}...)" if tcs else f"Abstenção: {a.get('content', '')[:60]}..."
                print(f"   [Nível {target_diff}] User: {u[:60]}... -> {tc_str}")
                break
