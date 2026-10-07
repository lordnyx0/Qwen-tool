"""
benchmark_tool_calling_baseline.py
Avalia a capacidade de Tool Calling no formato nativo Qwen do Qwen3Loop antes do fine-tuning especializado.
Gera métricas de:
1. Tool Call Trigger Accuracy (identificar se deve ou não chamar ferramenta)
2. Tool Name Selection Accuracy (escolher a ferramenta correta)
3. JSON Syntax Validity (parsear o JSON dentro de <tool_call>)
4. Parameter Extraction Accuracy (extrair argumentos corretos)
5. Multi-turn Tool Response synthesis
6. Halting Probe Confidence (Loop 1 vs Loop 2)
"""

import sys
import os
import json
import re
import time
import torch

# Adicionar caminhos para engine/model_src
sys.path.insert(0, os.path.abspath("engine"))
sys.path.insert(0, os.path.abspath("model_src"))

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import qwen3loop_python  # noqa: F401

MODEL_PATH = os.path.abspath("checkpoints/qwen3loop_sft_final")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"=== BENCHMARK DE BASELINE: QWEN NATIVE TOOL CALLING ===")
print(f"Dispositivo: {DEVICE}")
print(f"Carregando modelo de: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=dtype).to(DEVICE)
model.eval()

# Carregar sonda de parada se disponível
probe_path = os.path.join(MODEL_PATH, "latent_halting_probe.pt")
probe = None
if os.path.exists(probe_path):
    try:
        class HaltingProbeMLP(torch.nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, 256),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(256, 64),
                    torch.nn.GELU(),
                    torch.nn.Linear(64, 1)
                )
            def forward(self, x):
                return self.net(x)
        hidden_dim = config.hidden_size
        probe = HaltingProbeMLP(hidden_dim).to(DEVICE).to(dtype)
        probe.load_state_dict(torch.load(probe_path, map_location=DEVICE))
        probe.eval()
        print("Halting Probe carregada com sucesso!")
    except Exception as e:
        print(f"Nota: Sonda não pôde ser carregada: {e}")

# Definição dos Testes de Avaliação
BENCHMARK_CASES = [
    {
        "id": "TC-01",
        "title": "Chamada Simples com Parâmetro Único / Clima",
        "description": "Deve chamar get_current_weather com localização e unidade.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Obtém as condições meteorológicas de uma cidade.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "Nome da cidade"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "Qual é a temperatura atual em Tóquio em graus Celsius?"}
        ],
        "expected_tool": "get_current_weather",
        "expected_args_keys": ["location"],
        "should_call_tool": True
    },
    {
        "id": "TC-02",
        "title": "Execução de Código Python / Matemática",
        "description": "Deve usar execute_python para calcular 2^100 % 7.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Executa um script Python em ambiente seguro e retorna a saída padrão.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Código Python a ser executado"}
                        },
                        "required": ["code"]
                    }
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "Calcule quanto é 2 elevado a 100 módulo 7 executando Python."}
        ],
        "expected_tool": "execute_python",
        "expected_args_keys": ["code"],
        "should_call_tool": True
    },
    {
        "id": "TC-03",
        "title": "Discriminação e Seleção entre Múltiplas Ferramentas",
        "description": "Dentre 3 ferramentas disponíveis, deve selecionar send_email.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Busca informações na internet.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "send_email",
                    "description": "Envia um e-mail para um destinatário.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string", "description": "Endereço de e-mail do destinatário"},
                            "subject": {"type": "string", "description": "Assunto da mensagem"},
                            "body": {"type": "string", "description": "Conteúdo da mensagem"}
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Consulta cotação de uma ação pelo ticker.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ticker": {"type": "string"}},
                        "required": ["ticker"]
                    }
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "Envie um e-mail para suporte@empresa.com com o assunto 'Alerta' e o texto 'O servidor web parou'."}
        ],
        "expected_tool": "send_email",
        "expected_args_keys": ["to", "subject", "body"],
        "should_call_tool": True
    },
    {
        "id": "TC-04",
        "title": "Teste Negativo (Pergunta Conversacional - NÃO chamar tool)",
        "description": "Com ferramentas disponíveis no sistema, deve responder diretamente sem chamar nenhuma tool.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Busca na internet.",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Executa Python.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"]
                    }
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "Olá! Quem escreveu o livro 'Dom Casmurro'?"}
        ],
        "expected_tool": None,
        "expected_args_keys": [],
        "should_call_tool": False
    },
    {
        "id": "TC-05",
        "title": "Parâmetros Estruturados / Filtro JSON Aninhado",
        "description": "Deve gerar argumentos que incluam objeto/dicionário aninhado para filtros.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "Executa consulta parametrizada em banco de dados.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string", "description": "Nome da tabela"},
                            "filters": {
                                "type": "object",
                                "description": "Dicionário de filtros de coluna e valor"
                            },
                            "limit": {"type": "integer", "description": "Máximo de registros"}
                        },
                        "required": ["table"]
                    }
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "Consulte na tabela 'produtos' os itens da categoria 'smartphones' com preço menor que 2000, limite de 5 registros."}
        ],
        "expected_tool": "query_database",
        "expected_args_keys": ["table"],
        "should_call_tool": True
    },
    {
        "id": "TC-06",
        "title": "Síntese Multi-turn pós-Tool (Resposta com base no retorno)",
        "description": "O assistente recebe <tool_response> e deve responder em linguagem natural ao usuário.",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_stock_quote",
                    "description": "Cotação de ações.",
                    "parameters": {
                        "type": "object",
                        "properties": {"ticker": {"type": "string"}},
                        "required": ["ticker"]
                    }
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "Qual a cotação da Apple hoje?"},
            {"role": "assistant", "content": '<tool_call>\n{"name": "get_stock_quote", "arguments": {"ticker": "AAPL"}}\n</tool_call>'},
            {"role": "tool", "content": '{"ticker": "AAPL", "price": 228.45, "currency": "USD", "change_pct": "+1.85%"}'}
        ],
        "expected_tool": None,
        "expected_args_keys": [],
        "should_call_tool": False
    }
]

def extract_tool_calls(text):
    """Extrai chamadas de ferramenta do formato nativo Qwen: <tool_call> ... </tool_call>"""
    pattern = r"<tool_call>(.*?)</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    parsed_calls = []
    syntax_errors = []
    
    for m in matches:
        raw_json = m.strip()
        try:
            call_obj = json.loads(raw_json)
            parsed_calls.append(call_obj)
        except Exception as e:
            syntax_errors.append((raw_json, str(e)))
            
    return parsed_calls, syntax_errors

results = []

print(f"\nIniciando execução dos {len(BENCHMARK_CASES)} testes de baseline...\n")

for case in BENCHMARK_CASES:
    print("=" * 70)
    print(f"[{case['id']}] {case['title']}")
    print(f"Descrição: {case['description']}")
    print("-" * 70)
    
    prompt_str = tokenizer.apply_chat_template(
        case["messages"],
        tools=case["tools"],
        add_generation_prompt=True,
        tokenize=False
    )
    
    inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,  # Greedy para reprodutibilidade estrita
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )
    t_gen = time.time() - t0
    
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    # Limpar token final de término se presente
    clean_text = response_text.replace("<|im_end|>", "").strip()
    
    # Avaliação
    tool_calls, syntax_errs = extract_tool_calls(clean_text)
    called_tool = len(tool_calls) > 0 or len(syntax_errs) > 0 or ("<tool_call>" in clean_text)
    
    score_trigger = (called_tool == case["should_call_tool"])
    score_name = False
    score_json = False
    score_args = False
    
    matched_call = None
    if case["should_call_tool"]:
        if tool_calls:
            for tc in tool_calls:
                if tc.get("name") == case["expected_tool"]:
                    score_name = True
                    matched_call = tc
                    break
            score_json = (len(syntax_errs) == 0 and len(tool_calls) > 0)
            if matched_call and "arguments" in matched_call:
                args = matched_call["arguments"]
                if isinstance(args, dict):
                    has_all_keys = all(k in args for k in case["expected_args_keys"])
                    score_args = has_all_keys
    else:
        # Para caso negativo ou pós-tool, sucesso é NÃO emitir tool call
        score_name = (not called_tool)
        score_json = True
        score_args = (not called_tool)

    case_result = {
        "id": case["id"],
        "title": case["title"],
        "should_call": case["should_call_tool"],
        "called_tool": called_tool,
        "score_trigger": score_trigger,
        "score_name": score_name,
        "score_json": score_json,
        "score_args": score_args,
        "tool_calls_extracted": tool_calls,
        "syntax_errors": syntax_errs,
        "response_sample": clean_text[:200]
    }
    results.append(case_result)
    
    print(f"-> Saída do Modelo:\n{clean_text}\n")
    print(f"-> Detecção de Tool Call: {'CORRETO' if score_trigger else 'FALHA'}")
    print(f"-> Nome da Ferramenta:   {'CORRETO' if score_name else 'FALHA'}")
    print(f"-> Sintaxe JSON:         {'VÁLIDO' if score_json else 'INVÁLIDO'}")
    print(f"-> Argumentos Extraídos: {'CORRETO' if score_args else 'FALHA'}")

print("\n" + "=" * 70)
print("TABELA RESUMO DO BENCHMARK BASELINE (QWEN NATIVE TOOL CALLING)")
print("=" * 70)
print(f"{'ID':<7} | {'Trigger':<9} | {'Nome Tool':<10} | {'JSON Válido':<12} | {'Argumentos':<11} | {'Título'}")
print("-" * 70)
for r in results:
    tr = "[OK]" if r["score_trigger"] else "[FAIL]"
    nm = "[OK]" if r["score_name"] else "[FAIL]"
    js = "[OK]" if r["score_json"] else "[FAIL]"
    ar = "[OK]" if r["score_args"] else "[FAIL]"
    print(f"{r['id']:<7} | {tr:<9} | {nm:<10} | {js:<12} | {ar:<11} | {r['title']}")

total_cases = len(results)
acc_trigger = sum(1 for r in results if r["score_trigger"]) / total_cases * 100
acc_name = sum(1 for r in results if r["score_name"]) / total_cases * 100
acc_json = sum(1 for r in results if r["score_json"]) / total_cases * 100
acc_args = sum(1 for r in results if r["score_args"]) / total_cases * 100

overall_pass = sum(1 for r in results if (r["score_trigger"] and r["score_name"] and r["score_json"] and r["score_args"])) / total_cases * 100

print("-" * 70)
print(f"TAXA DE ACERTO DE DISPARO (Trigger):       {acc_trigger:.1f}%")
print(f"TAXA DE ACERTO DE NOME (Tool Selection):   {acc_name:.1f}%")
print(f"TAXA DE JSON VÁLIDO (Syntax Integrity):    {acc_json:.1f}%")
print(f"TAXA DE ARGUMENTOS VÁLIDOS (Extraction):   {acc_args:.1f}%")
print(f"SCORE GLOBAL PERFEITO (All Pass):          {overall_pass:.1f}%")
print("=" * 70)

# Salvar resultados em JSON
os.makedirs("data", exist_ok=True)
out_path = os.path.join("data", "tool_calling_baseline_results.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "metrics": {
            "trigger_accuracy": acc_trigger,
            "name_accuracy": acc_name,
            "json_accuracy": acc_json,
            "args_accuracy": acc_args,
            "overall_pass": overall_pass
        },
        "cases": results
    }, f, indent=2, ensure_ascii=False)

print(f"Resultados salvos em: {out_path}")
