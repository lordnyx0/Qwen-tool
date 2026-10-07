"""
build_sft_tool_dataset.py
Constrói o dataset supervisionado (SFT) especializado em Tool Calling para o Qwen3Loop-0.6B:
- 100% focado em Tool Calling e Abstenção (sem âncoras de texto puro)
- Nível 1: Foundation (Chamadas simples de 1 tool) + Negative Abstention (abstenção com resposta direta)
- Nível 2: Discrimination (Seleção correta entre múltiplas tools concorrentes com schemas complexos)
- Nível 3: Algorithmic & Code Execution (execute_python para matemática exata, modular, primos, fibonacci, etc.)
- Exclusão estrita das 90 amostras do benchmark de holdout (eval_benchmark_battery.jsonl)
- Raciocínio conciso (<think>) para cada chamada, alinhado ao chat_template.jinja do Qwen3Loop
"""

import os
import json
import random
import math

random.seed(42)

ROOT_DIR = os.path.abspath(".")
CURRICULUM_DIR = os.path.join(ROOT_DIR, "data", "rl_curriculum")
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "sft_tool_calling_curated.jsonl")

PYTHON_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": "Executa código Python em ambiente isolado e retorna o valor de saída padrão (stdout) ou resultado.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Código Python válido a ser executado."
                }
            },
            "required": ["code"]
        }
    }
}

# 1. Carregar IDs do Benchmark Holdout para NUNCA vazar no treino
holdout_ids = set()
eval_battery_path = os.path.join(CURRICULUM_DIR, "eval_benchmark_battery.jsonl")
if os.path.exists(eval_battery_path):
    with open(eval_battery_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                holdout_ids.add(item.get("id"))
print(f"Holdout benchmark: {len(holdout_ids)} IDs protegidos contra vazamento.")

sft_samples = []

# ---------------------------------------------------------
# FASE 1: NÍVEL 1 - FOUNDATION & ABSTENTION
# ---------------------------------------------------------
l1_path = os.path.join(CURRICULUM_DIR, "level_1_foundation.jsonl")
print(f"Processando {l1_path}...")
l1_count = 0
with open(l1_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("id") in holdout_ids:
            continue

        cat = item.get("category", "")
        tools = item.get("tools", [])
        msgs = item.get("messages", [])
        user_content = msgs[0]["content"] if msgs else ""
        expected_call = item.get("expected_call")
        expected_resp = item.get("expected_response", "")

        if expected_call:
            t_name = expected_call.get("name", "tool")
            reasoning = f"The user is requesting assistance with '{user_content[:60]}...'. I will call '{t_name}' to fulfill this request."
            asst_msg = {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": expected_call
                }]
            }
        else:
            # Abstenção
            if not expected_resp:
                expected_resp = "I am sorry, but I do not have a tool available to perform this specific request."
            reasoning = f"The user asks: '{user_content[:60]}...'. Looking at the available tools, none can perform this action. I should decline politely and answer directly without calling any tools."
            asst_msg = {
                "role": "assistant",
                "reasoning_content": reasoning,
                "content": expected_resp,
                "tool_calls": []
            }

        sft_samples.append({
            "id": item.get("id", f"l1_{l1_count}"),
            "category": cat,
            "difficulty": 1,
            "tools": tools,
            "messages": [
                {"role": "user", "content": user_content},
                asst_msg
            ]
        })
        l1_count += 1

print(f"Nível 1 adicionado: {l1_count} amostras.")

# ---------------------------------------------------------
# FASE 2: NÍVEL 2 - DISCRIMINATION & COMPLEX SCHEMAS
# ---------------------------------------------------------
l2_path = os.path.join(CURRICULUM_DIR, "level_2_discrimination.jsonl")
print(f"Processando {l2_path}...")
l2_count = 0
with open(l2_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("id") in holdout_ids:
            continue

        tools = item.get("tools", [])
        msgs = item.get("messages", [])
        user_content = msgs[0]["content"] if msgs else ""
        expected_call = item.get("expected_call")
        if not expected_call:
            continue

        t_name = expected_call.get("name", "")
        args = expected_call.get("arguments", {})
        arg_keys = ", ".join(args.keys()) if isinstance(args, dict) else ""
        reasoning = f"Evaluating user request against available functions. The best matching function is '{t_name}' with parameters ({arg_keys})."

        asst_msg = {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": expected_call
            }]
        }

        sft_samples.append({
            "id": item.get("id", f"l2_{l2_count}"),
            "category": item.get("category", "discrimination"),
            "difficulty": 2,
            "tools": tools,
            "messages": [
                {"role": "user", "content": user_content},
                asst_msg
            ]
        })
        l2_count += 1

print(f"Nível 2 adicionado: {l2_count} amostras.")

# ---------------------------------------------------------
# FASE 3: NÍVEL 3 - PYTHON EXECUTION & ALGORITHMIC MATH
# ---------------------------------------------------------
l3_path = os.path.join(CURRICULUM_DIR, "level_3_reasoning_python.jsonl")
print(f"Processando {l3_path}...")
l3_count = 0
with open(l3_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("id") in holdout_ids:
            continue

        tools = item.get("tools", [PYTHON_TOOL_SCHEMA])
        msgs = item.get("messages", [])
        user_content = msgs[0]["content"] if msgs else ""
        expected_call = item.get("expected_call")
        if not expected_call:
            continue

        code = expected_call.get("arguments", {}).get("code", "")
        reasoning = f"Para resolver este problema matemático/computacional com exatidão e sem erros de aproximação, utilizarei o interpretador Python via execute_python."

        asst_msg = {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "arguments": {"code": code}
                }
            }]
        }

        sft_samples.append({
            "id": item.get("id", f"l3_{l3_count}"),
            "category": "python_math_execution",
            "difficulty": 3,
            "tools": tools,
            "messages": [
                {"role": "user", "content": user_content},
                asst_msg
            ]
        })
        l3_count += 1

print(f"Nível 3 existente adicionado: {l3_count} amostras.")

# Gerar amostras sintéticas extras de Nível 3 para totalizar ~1.000 amostras de Python Math
extra_needed = 1000 - l3_count
if extra_needed > 0:
    print(f"Gerando {extra_needed} amostras sintéticas adicionais de Python Math (Nível 3)...")
    for i in range(extra_needed):
        ptype = i % 8
        if ptype == 0:
            base = random.randint(2, 9)
            exp = random.randint(50, 600)
            mod = random.choice([7, 11, 13, 17, 19, 23, 29, 31, 37, 41])
            q = f"Calcule exatamente o valor de {base}^{exp} mod {mod} executando Python."
            code = f"print(pow({base}, {exp}, {mod}))"
            reasoning = f"Cálculo de exponenciação modular com números grandes ({base}^{exp} mod {mod}). Executarei pow({base}, {exp}, {mod}) via Python."
        elif ptype == 1:
            limit = random.choice([50, 100, 150, 200, 250, 300])
            q = f"Encontre a soma de todos os números primos menores que {limit} usando código Python."
            code = f"def is_p(n): return n > 1 and all(n % d != 0 for d in range(2, int(n**0.5) + 1))\nprint(sum(n for n in range({limit}) if is_p(n)))"
            reasoning = f"Determinação de números primos até {limit} e cálculo da soma total via teste de primalidade em Python."
        elif ptype == 2:
            n = random.randint(20, 60)
            q = f"Calcule o fatorial de {n} e informe o número total de dígitos usando Python."
            code = f"import math\nf = math.factorial({n})\nprint(len(str(f)))"
            reasoning = f"Cálculo do fatorial {n}! e contagem de dígitos da string resultante usando math.factorial."
        elif ptype == 3:
            a = random.randint(500, 50000)
            b = random.randint(500, 50000)
            q = f"Calcule o Máximo Divisor Comum (MDC) e Mínimo Múltiplo Comum (MMC) entre {a} e {b} via script Python."
            code = f"import math\ng = math.gcd({a}, {b})\nl = abs({a} * {b}) // g\nprint(f'MDC={{g}}, MMC={{l}}')"
            reasoning = f"Cálculo de MDC e MMC entre {a} e {b} usando math.gcd em Python."
        elif ptype == 4:
            n = random.randint(15, 35)
            k = random.randint(3, 8)
            q = f"Quantas combinações simples de {n} elementos tomados {k} a {k} existem? Use Python para calcular C({n}, {k})."
            code = f"import math\nprint(math.comb({n}, {k}))"
            reasoning = f"Cálculo combinatório C({n}, {k}) através da biblioteca padrão math.comb do Python."
        elif ptype == 5:
            n = random.randint(25, 70)
            q = f"Calcule o {n}-ésimo termo da sequência de Fibonacci usando Python."
            code = f"a, b = 0, 1\nfor _ in range({n}): a, b = b, a + b\nprint(a)"
            reasoning = f"Geração do {n}-ésimo elemento da sequência de Fibonacci iterativamente em Python."
        elif ptype == 6:
            lst = [random.randint(1, 1000) for _ in range(12)]
            q = f"Dada a lista {lst}, encontre a média aritmética e o desvio padrão amostral executando Python."
            code = f"import statistics\ndata = {lst}\nprint(f'Média={{statistics.mean(data):.2f}}, Desvio={{statistics.stdev(data):.2f}}')"
            reasoning = f"Cálculo estatístico de média e desvio padrão amostral para a lista fornecida via módulo statistics."
        else:
            text = random.choice([
                "O rato roeu a roupa do rei de Roma",
                "Python e inteligência artificial para automação de tarefas",
                "Estruturas de dados avançadas e algoritmos recorrentes"
            ])
            q = f"Conte a frequência de cada caractere na frase '{text}' ignorando espaços via Python."
            code = f"from collections import Counter\ns = '{text}'.replace(' ', '').lower()\nprint(dict(Counter(s)))"
            reasoning = f"Contagem e análise de frequência de caracteres via collections.Counter em Python."

        asst_msg = {
            "role": "assistant",
            "reasoning_content": reasoning,
            "content": "",
            "tool_calls": [{
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "arguments": {"code": code}
                }
            }]
        }

        sft_samples.append({
            "id": f"py_math_synth_{i+1:04d}",
            "category": "python_math_execution",
            "difficulty": 3,
            "tools": [PYTHON_TOOL_SCHEMA],
            "messages": [
                {"role": "user", "content": q},
                asst_msg
            ]
        })
        l3_count += 1

print(f"Total Nível 3: {l3_count} amostras.")

# Embaralhar todas as amostras
random.shuffle(sft_samples)

print("\n" + "=" * 65)
print(f"SALVANDO DATASET SFT TOOL CALLING EM: {OUTPUT_PATH}")
print("=" * 65)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for s in sft_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"Sucesso! Total de amostras SFT geradas: {len(sft_samples):,}")
print(f"-> Nível 1: {l1_count:,}")
print(f"-> Nível 2: {l2_count:,}")
print(f"-> Nível 3: {l3_count:,}")
