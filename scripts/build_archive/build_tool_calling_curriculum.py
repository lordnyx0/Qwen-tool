"""
build_tool_calling_curriculum.py
Constrói o dataset de Curriculum Learning para Tool Calling no formato nativo Qwen:
- Level 1: Foundation & Abstention (500 chamadas simples de 1 tool + 500 abstenções conversacionais sem tool)
- Level 2: Discrimination & Structured Schemas (1.200 chamadas multi-ferramentas concorrentes e filtros aninhados)
- Level 3: Algorithmic & Python Execution (600 problemas de matemática/código com execute_python + 200 APIs complexas)
- Eval Benchmark: 200 amostras holdout balanceadas para teste durante o treino
"""

import os
import json
import re
import random
from datasets import load_dataset

random.seed(42)

OUTPUT_DIR = os.path.abspath("data/rl_curriculum")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== CONSTRUÇÃO COMPLETA DO DATASET DE TOOL CALLING COM CURRICULUM ===")

# -----------------------------------------------------------------
# 1. Amostras Sintéticas de Nível 3: Python Execution (Matemática e Algoritmos)
# -----------------------------------------------------------------
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

def generate_python_math_samples(count=650):
    samples = []
    for i in range(count):
        ptype = i % 6
        if ptype == 0:
            base = random.randint(2, 9)
            exp = random.randint(50, 500)
            mod = random.choice([7, 11, 13, 17, 19, 23, 29])
            query = f"Calcule exatamente o valor de {base}^{exp} mod {mod} executando Python."
            code = f"print(pow({base}, {exp}, {mod}))"
            ans = str(pow(base, exp, mod))
        elif ptype == 1:
            limit = random.choice([50, 100, 150, 200, 300])
            query = f"Encontre a soma de todos os números primos menores que {limit} usando código Python."
            code = f"def is_p(n): return n > 1 and all(n % d != 0 for d in range(2, int(n**0.5) + 1))\nprint(sum(n for n in range({limit}) if is_p(n)))"
            def is_p(n): return n > 1 and all(n % d != 0 for d in range(2, int(n**0.5) + 1))
            ans = str(sum(n for n in range(limit) if is_p(n)))
        elif ptype == 2:
            n = random.randint(15, 50)
            query = f"Calcule o fatorial de {n} e informe o número total de dígitos usando Python."
            code = f"import math\nf = math.factorial({n})\nprint(len(str(f)))"
            import math
            ans = str(len(str(math.factorial(n))))
        elif ptype == 3:
            a = random.randint(100, 10000)
            b = random.randint(100, 10000)
            query = f"Calcule o Máximo Divisor Comum (MDC) e Mínimo Múltiplo Comum (MMC) entre {a} e {b} via script Python."
            code = f"import math\ng = math.gcd({a}, {b})\nl = abs({a} * {b}) // g\nprint(f'MDC={{g}}, MMC={{l}}')"
            import math
            g = math.gcd(a, b)
            l = abs(a * b) // g
            ans = f"MDC={g}, MMC={l}"
        elif ptype == 4:
            n = random.randint(12, 30)
            k = random.randint(3, 8)
            query = f"Quantas combinações simples de {n} elementos tomados {k} a {k} existem? Use Python para calcular C({n}, {k})."
            code = f"import math\nprint(math.comb({n}, {k}))"
            import math
            ans = str(math.comb(n, k))
        else:
            n = random.randint(20, 60)
            query = f"Calcule o {n}-ésimo termo da sequência de Fibonacci usando Python."
            code = f"a, b = 0, 1\nfor _ in range({n}): a, b = b, a + b\nprint(a)"
            a, b = 0, 1
            for _ in range(n): a, b = b, a + b
            ans = str(a)

        samples.append({
            "id": f"py_math_{i+1:04d}",
            "difficulty": 3,
            "category": "python_math_execution",
            "tools": [PYTHON_TOOL_SCHEMA],
            "messages": [
                {"role": "user", "content": query}
            ],
            "expected_call": {
                "name": "execute_python",
                "arguments": {"code": code}
            },
            "expected_execution_result": ans
        })
    return samples

# -----------------------------------------------------------------
# 2. Mineração do Hermes Function Calling (Nível 2 & Nível 3)
# -----------------------------------------------------------------
print("Carregando e processando Hermes Function Calling...")
ds_hermes = load_dataset("NousResearch/hermes-function-calling-v1", split="train")

hermes_level_2 = []
hermes_level_3 = []

def parse_hermes_tool_call(convs):
    calls = []
    for c in convs:
        if c.get("from") in ["gpt", "assistant"] or c.get("role") == "assistant":
            val = c.get("value", "")
            matches = re.findall(r"<tool_call>(.*?)</tool_call>", val, re.DOTALL)
            for m in matches:
                try:
                    calls.append(json.loads(m.strip()))
                except Exception:
                    pass
    return calls

for idx, row in enumerate(ds_hermes):
    tools = row.get("tools", [])
    convs = row.get("conversations", [])
    if not tools or len(convs) < 2:
        continue

    user_msg = None
    for c in convs:
        if c.get("from") in ["human", "user"] or c.get("role") == "user":
            user_msg = c.get("value", "")
            break
    if not user_msg:
        continue

    calls = parse_hermes_tool_call(convs)
    if not calls:
        continue

    call = calls[0]
    c_name = call.get("name", "")
    args = call.get("arguments", {})

    sample_obj = {
        "id": f"hermes_{idx:05d}",
        "tools": tools,
        "messages": [{"role": "user", "content": user_msg}],
        "expected_call": call,
        "category": row.get("category", "general")
    }

    has_code_or_sql = any(k in c_name.lower() for k in ["python", "code", "sql", "bash", "execute", "eval", "query"])
    if has_code_or_sql or len(convs) > 4:
        sample_obj["difficulty"] = 3
        hermes_level_3.append(sample_obj)
    else:
        sample_obj["difficulty"] = 2
        hermes_level_2.append(sample_obj)

print(f"Hermes processado -> L2: {len(hermes_level_2)} | L3: {len(hermes_level_3)}")

# -----------------------------------------------------------------
# 3. Mineração do Glaive Function Calling v2 (Nível 1 Simples & Negativos)
# -----------------------------------------------------------------
print("Carregando e processando Glaive Function Calling v2 (L1 Simples e Abstenção)...")
ds_glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)

glaive_level_1_simple = []
glaive_level_1_negative = []

def parse_glaive_tools_from_system(sys_text):
    """Extrai os schemas de ferramenta da string de system do Glaive."""
    tools = []
    # No Glaive, funções são blocos JSON no system após 'following functions. Use them if required -\n'
    idx = sys_text.find("following functions. Use them if required -")
    if idx != -1:
        json_part = sys_text[idx + len("following functions. Use them if required -"):].strip()
        # Encontrar blocos JSON
        raw_blocks = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", json_part, re.DOTALL)
        for b in raw_blocks:
            try:
                data = json.loads(b)
                if "name" in data:
                    tools.append({
                        "type": "function",
                        "function": data
                    })
            except Exception:
                pass
    return tools

def parse_glaive_function_call(chat_text):
    """Extrai <functioncall> {name, arguments} do chat do Glaive."""
    import ast
    m = re.search(r"<functioncall>\s*(\{.*?\})\s*(?:<\|endoftext\|>|\n|$)", chat_text, re.DOTALL)
    if m:
        raw = m.group(1).strip()
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed.get("arguments"), str):
                try:
                    parsed["arguments"] = json.loads(parsed["arguments"])
                except Exception:
                    pass
            return parsed
        except Exception:
            pass
        # Fallback por regex
        m_name = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', raw)
        m_args = re.search(r'["\']arguments["\']\s*:\s*([\'"].*?[\'"])\s*\}', raw, re.DOTALL)
        if m_name and m_args:
            name = m_name.group(1)
            raw_args = m_args.group(1).strip("'\"")
            try:
                args = json.loads(raw_args)
            except Exception:
                args = {}
            return {"name": name, "arguments": args}
    return None

for idx, row in enumerate(ds_glaive):
    chat = row.get("chat", "")
    sys = row.get("system", "")

    # Extrair usuário
    user_m = re.search(r"USER:\s*(.*?)(?=\n\s*ASSISTANT:)", chat, re.DOTALL)
    if not user_m:
        continue
    user_query = user_m.group(1).strip()
    if len(user_query) < 5:
        continue

    tools = parse_glaive_tools_from_system(sys)
    if not tools:
        continue

    # Caso 1: Chamada Positiva Simples (1 tool, <= 2 parâmetros primitivos)
    if "<functioncall>" in chat and len(glaive_level_1_simple) < 600:
        call = parse_glaive_function_call(chat)
        if call and "name" in call:
            args = call.get("arguments", {})
            if isinstance(args, dict) and len(args) <= 2:
                if all(isinstance(v, (str, int, float, bool)) for v in args.values()):
                    glaive_level_1_simple.append({
                        "id": f"glaive_l1_{len(glaive_level_1_simple):04d}",
                        "difficulty": 1,
                        "category": "single_tool_simple",
                        "tools": tools[:1],
                        "messages": [{"role": "user", "content": user_query}],
                        "expected_call": call
                    })

    # Caso 2: Chamada Negativa (Abstenção: tem tools, mas responde direto)
    elif "<functioncall>" not in chat and len(glaive_level_1_negative) < 600:
        asst_m = re.search(r"ASSISTANT:\s*(.*?)(?=<\|endoftext\|>|\n\s*USER:|$)", chat, re.DOTALL)
        if asst_m:
            asst_resp = asst_m.group(1).strip()
            if len(asst_resp) > 10:
                glaive_level_1_negative.append({
                    "id": f"glaive_neg_{len(glaive_level_1_negative):04d}",
                    "difficulty": 1,
                    "category": "negative_abstention",
                    "tools": tools,
                    "messages": [{"role": "user", "content": user_query}],
                    "expected_call": None,
                    "expected_response": asst_resp[:200]
                })

    if len(glaive_level_1_simple) >= 600 and len(glaive_level_1_negative) >= 600:
        break

print(f"Glaive processado -> L1 Simples: {len(glaive_level_1_simple)} | L1 Negativos: {len(glaive_level_1_negative)}")

# -----------------------------------------------------------------
# 4. Perguntas Conversacionais Cotidianas Extras para Abstenção
# -----------------------------------------------------------------
EXTRA_CONVERSATIONAL = [
    "Olá! Como você está hoje?",
    "Quem escreveu o romance 'Dom Casmurro'?",
    "O que é a fotossíntese e por que ela é importante para a vida na Terra?",
    "Me conte uma curiosidade histórica interessante sobre o Egito Antigo.",
    "Qual é a diferença entre um átomo e uma molécula?",
    "Escreva uma mensagem de bom dia curta e inspiradora para começar a semana.",
    "O que significa a expressão 'dar com os burros n'água'?",
    "Explique de forma simples o conceito de gravidade para uma criança de 8 anos.",
    "Qual é a capital do Canadá?",
    "Quantos minutos tem um dia completo?"
]

extra_negatives = []
for i, q in enumerate(EXTRA_CONVERSATIONAL):
    extra_negatives.append({
        "id": f"extra_neg_{i+1:03d}",
        "difficulty": 1,
        "category": "conversational_abstention",
        "tools": [PYTHON_TOOL_SCHEMA],
        "messages": [{"role": "user", "content": q}],
        "expected_call": None
    })

# -----------------------------------------------------------------
# 5. Montagem, Balanceamento e Salvamento das 3 Fases
# -----------------------------------------------------------------
random.shuffle(glaive_level_1_simple)
random.shuffle(glaive_level_1_negative)
random.shuffle(hermes_level_2)
random.shuffle(hermes_level_3)

py_math_samples = generate_python_math_samples(count=650)

# FASE 1: 500 chamadas simples + 500 abstenções (Glaive + extras) = 1.000 prompts
l1_train = glaive_level_1_simple[:500] + glaive_level_1_negative[:490] + extra_negatives[:10]
random.shuffle(l1_train)

# FASE 2: 1.200 prompts de discriminação multi-tools e filtros aninhados
l2_train = hermes_level_2[:1200]
random.shuffle(l2_train)

# FASE 3: 600 python execution math + 200 hermes complexos = 800 prompts
l3_train = py_math_samples[:600] + hermes_level_3[:200]
random.shuffle(l3_train)

# EVAL BENCHMARK (Holdout): 200 prompts (60 L1, 80 L2, 60 L3)
eval_benchmark = (
    glaive_level_1_simple[500:530] + glaive_level_1_negative[490:520] +  # 60 L1
    hermes_level_2[1200:1280] +                                         # 80 L2
    py_math_samples[600:630] + hermes_level_3[200:230]                  # 60 L3
)
random.shuffle(eval_benchmark)

def save_jsonl(filepath, data):
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Salvo: {filepath} ({len(data)} amostras)")

save_jsonl(os.path.join(OUTPUT_DIR, "level_1_foundation.jsonl"), l1_train)
save_jsonl(os.path.join(OUTPUT_DIR, "level_2_discrimination.jsonl"), l2_train)
save_jsonl(os.path.join(OUTPUT_DIR, "level_3_reasoning_python.jsonl"), l3_train)
save_jsonl(os.path.join(OUTPUT_DIR, "eval_benchmark_battery.jsonl"), eval_benchmark)

print("\n" + "=" * 65)
print(f"CURRICULUM DATASET PRONTO PARA TREINO RL/GRPO:")
print(f"-> Nível 1 (Fundação & Abstenção):    {len(l1_train)} prompts")
print(f"-> Nível 2 (Discriminação & Schemas): {len(l2_train)} prompts")
print(f"-> Nível 3 (Python Math & Algoritmo): {len(l3_train)} prompts")
print(f"-> Benchmark Hold-out de Avaliação:   {len(eval_benchmark)} prompts")
print(f"-> TOTAL GERAL:                       {len(l1_train) + len(l2_train) + len(l3_train) + len(eval_benchmark)} prompts")
print("=" * 65)
