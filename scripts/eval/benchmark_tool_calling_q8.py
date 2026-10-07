"""
benchmark_tool_calling_q8.py
Executa a bateria de benchmark de Tool Calling no formato nativo Qwen
utilizando os modelos quantizados GGUF Q8 via llama-cpp-python com aceleração GPU CUDA:
1. models/gguf/unrolled_modelo_qwen3loop_sft_q8_0.gguf (Nosso modelo Qwen3Loop 56L Q8)
2. models/gguf/Qwen3-0.6B-Q8_0.gguf (Modelo Base oficial Qwen 28L Q8)
"""

import os
import sys
import re
import json
import time
from llama_cpp import Llama
from transformers import AutoTokenizer

MODEL_GRPO_Q8 = os.path.abspath("models/gguf/unrolled_modelo_qwen3loop_grpo_q8_0.gguf")
MODEL_QWEN3LOOP_Q8 = os.path.abspath("models/gguf/unrolled_modelo_qwen3loop_sft_q8_0.gguf")
MODEL_BASE_Q8 = os.path.abspath("models/gguf/Qwen3-0.6B-Q8_0.gguf")
TOKENIZER_DIR = os.path.abspath("checkpoints/qwen3loop_sft_final")


print("=== BENCHMARK DE TOOL CALLING EM MODELOS GGUF Q8 (CUDA) ===")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

BENCHMARK_CASES = [
    {
        "id": "TC-01",
        "title": "Chamada Simples (Clima: get_current_weather)",
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
        "expected_args": ["location"],
        "should_call": True
    },
    {
        "id": "TC-02",
        "title": "Execução Python / Matemática (execute_python)",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Executa script Python e retorna a saída padrão.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string"}},
                        "required": ["code"]
                    }
                }
            }
        ],
        "messages": [
            {"role": "user", "content": "Calcule quanto é 2 elevado a 100 módulo 7 executando Python."}
        ],
        "expected_tool": "execute_python",
        "expected_args": ["code"],
        "should_call": True
    },
    {
        "id": "TC-03",
        "title": "Discriminação entre Múltiplas Ferramentas (send_email)",
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
                    "description": "Envia um e-mail.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "to": {"type": "string"},
                            "subject": {"type": "string"},
                            "body": {"type": "string"}
                        },
                        "required": ["to", "subject", "body"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_stock_price",
                    "description": "Consulta cotação de ação.",
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
        "expected_args": ["to", "subject", "body"],
        "should_call": True
    },
    {
        "id": "TC-04",
        "title": "Teste Negativo (Pergunta Conversacional - Abstenção)",
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
        "expected_args": [],
        "should_call": False
    },
    {
        "id": "TC-05",
        "title": "Parâmetros Estruturados / Filtro JSON Aninhado",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "query_database",
                    "description": "Executa consulta em banco de dados.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "table": {"type": "string"},
                            "filters": {"type": "object"},
                            "limit": {"type": "integer"}
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
        "expected_args": ["table"],
        "should_call": True
    },
    {
        "id": "TC-06",
        "title": "Síntese Multi-turn pós-Tool (Resposta com retorno)",
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
        "expected_args": [],
        "should_call": False
    }
]

def extract_tool_calls(text):
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    parsed = []
    errors = []
    for m in matches:
        try:
            obj = json.loads(m.strip())
            if isinstance(obj, dict):
                parsed.append(obj)
        except Exception as e:
            errors.append((m.strip(), str(e)))
    return parsed, errors

def evaluate_model_gguf(model_path: str, model_name: str):
    print("\n" + "=" * 75)
    print(f"AVALIANDO MODELO: {model_name}")
    print(f"Caminho: {model_path}")
    print("=" * 75)
    
    if not os.path.exists(model_path):
        print(f"ERRO: Arquivo {model_path} não encontrado!")
        return None

    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False
    )
    
    results = []
    
    for case in BENCHMARK_CASES:
        prompt_str = tokenizer.apply_chat_template(
            case["messages"],
            tools=case["tools"],
            add_generation_prompt=True,
            tokenize=False
        )
        
        t0 = time.time()
        res = llm(
            prompt_str,
            max_tokens=250,
            temperature=0.0,  # Greedy
            repeat_penalty=1.05,
            stop=["<|im_end|>"]
        )
        gen_time = time.time() - t0
        text = res["choices"][0]["text"].strip()
        n_toks = res["usage"]["completion_tokens"]
        tok_per_sec = n_toks / max(1e-4, gen_time)
        
        tool_calls, syntax_errs = extract_tool_calls(text)
        called = len(tool_calls) > 0 or len(syntax_errs) > 0 or ("<tool_call>" in text)
        
        score_trigger = (called == case["should_call"])
        score_name = False
        score_json = False
        score_args = False
        
        matched_call = None
        if case["should_call"]:
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
                        score_args = all(k in args for k in case["expected_args"])
        else:
            score_name = (not called)
            score_json = True
            score_args = (not called)
            
        case_res = {
            "id": case["id"],
            "title": case["title"],
            "trigger": score_trigger,
            "name": score_name,
            "json": score_json,
            "args": score_args,
            "pass": (score_trigger and score_name and score_json and score_args),
            "speed": tok_per_sec,
            "text": text[:150]
        }
        results.append(case_res)
        
        status_str = "[PASS]" if case_res["pass"] else "[FAIL]"
        print(f"[{case['id']}] {status_str} (Trigger={score_trigger}, Nome={score_name}, JSON={score_json}, Args={score_args}) - {tok_per_sec:.1f} tok/s")
        print(f"       Saída: {text[:100]}...\n")

    total = len(results)
    pass_cnt = sum(1 for r in results if r["pass"])
    tr_acc = sum(1 for r in results if r["trigger"]) / total * 100
    nm_acc = sum(1 for r in results if r["name"]) / total * 100
    js_acc = sum(1 for r in results if r["json"]) / total * 100
    ar_acc = sum(1 for r in results if r["args"]) / total * 100
    avg_speed = sum(r["speed"] for r in results) / total
    
    print("-" * 75)
    print(f"RESUMO: {model_name}")
    print(f"Taxa de Acerto Geral (All Pass): {pass_cnt}/{total} ({pass_cnt/total*100:.1f}%)")
    print(f"Trigger: {tr_acc:.1f}% | Nome: {nm_acc:.1f}% | JSON: {js_acc:.1f}% | Args: {ar_acc:.1f}%")
    print(f"Velocidade Média: {avg_speed:.1f} tokens/s")
    print("-" * 75)
    
    return {
        "model_name": model_name,
        "results": results,
        "pass_rate": pass_cnt / total * 100,
        "trigger_acc": tr_acc,
        "name_acc": nm_acc,
        "json_acc": js_acc,
        "args_acc": ar_acc,
        "avg_speed": avg_speed
    }

def main():
    # 1. Avaliar o modelo GRPO Q8 treinado (56 Camadas)
    grpo_res = evaluate_model_gguf(MODEL_GRPO_Q8, "Qwen3Loop-GRPO-56L-Q8_0 (Novo Treinado)")

    # 2. Avaliar nosso Qwen3Loop SFT Q8 (56 Camadas)
    qwen3loop_res = evaluate_model_gguf(MODEL_QWEN3LOOP_Q8, "Qwen3Loop-SFT-56L-Q8_0 (Base SFT)")
    
    # 3. Avaliar o Base oficial Qwen 28L Q8 para comparação
    base_res = evaluate_model_gguf(MODEL_BASE_Q8, "Qwen3-Base-Official-28L-Q8_0 (Oficial)")
    
    # Salvar resultados
    out_file = os.path.abspath("data/tool_calling_q8_benchmark_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "qwen3loop_grpo_q8": grpo_res,
            "qwen3loop_56l_q8": qwen3loop_res,
            "qwen3_base_28l_q8": base_res
        }, f, indent=2, ensure_ascii=False)
    print(f"\nResultados completos salvos em: {out_file}")


if __name__ == "__main__":
    main()

