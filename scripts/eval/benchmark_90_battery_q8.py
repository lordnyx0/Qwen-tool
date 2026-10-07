"""
benchmark_90_battery_q8.py
Executa a bateria de avaliação hold-out de 90 itens (eval_benchmark_battery.jsonl)
nos modelos quantizados GGUF Q8 via llama-cpp-python com aceleração GPU (CUDA).

Modelos suportados:
  1. unrolled_sft: models/gguf/unrolled_modelo_qwen3loop_sft_q8_0.gguf (Base do treino GRPO)
  2. base: models/gguf/Qwen3-0.6B-Q8_0.gguf (Base oficial Qwen3-0.6B)
"""

import os
import sys
import json
import time
import argparse
import subprocess
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from llama_cpp import Llama
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "eval_legacy", "eval_benchmark_battery_90.jsonl")
TOOL_TOKENIZER_DIR = os.path.join(ROOT_DIR, "checkpoints", "qwen3loop_tool_sft_final")
TOKENIZER_DIR = TOOL_TOKENIZER_DIR if os.path.exists(os.path.join(TOOL_TOKENIZER_DIR, "tokenizer.json")) else os.path.join(ROOT_DIR, "checkpoints", "qwen3loop_sft_final")

MODELS = {
    "tool_sft_q8": {
        "name": "Qwen3Loop-Tool-SFT Q8 (2 Épocas - Final)",
        "path": os.path.join(ROOT_DIR, "models", "gguf", "unrolled_modelo_qwen3loop_tool_sft_q8_0.gguf")
    },
    "tool_sft_ckpt150_q8": {
        "name": "Qwen3Loop-Tool-SFT Q8 (1 Época - Ckpt 150)",
        "path": os.path.join(ROOT_DIR, "models", "gguf", "unrolled_modelo_qwen3loop_tool_sft_ckpt150_q8_0.gguf")
    },
    "grpo_q8": {
        "name": "Qwen3Loop-GRPO Q8 (Final Treinado)",
        "path": os.path.join(ROOT_DIR, "models", "gguf", "unrolled_modelo_qwen3loop_grpo_q8_0.gguf")
    },
    "unrolled_sft": {
        "name": "Qwen3Loop-SFT Q8 (Base do GRPO)",
        "path": os.path.join(ROOT_DIR, "models", "gguf", "unrolled_modelo_qwen3loop_sft_q8_0.gguf")
    },
    "base": {
        "name": "Qwen3-0.6B Base Q8 (Oficial)",
        "path": os.path.join(ROOT_DIR, "models", "gguf", "Qwen3-0.6B-Q8_0.gguf")
    }
}



def extract_tool_calls_from_text(text: str) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
    """Extrai chamadas de ferramenta do formato nativo Qwen com suporte a fallbacks."""
    import re
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    parsed = []
    errors = []
    has_xml_tag = ("<tool_call>" in text)

    # Se a tag não foi fechada mas existe no texto, tenta extrair a partir de <tool_call>
    if not matches and "<tool_call>" in text:
        after_tag = text.split("<tool_call>", 1)[1]
        matches = [after_tag]

    for m in matches:
        clean_json = m.strip()
        if "</tool_call>" in clean_json:
            clean_json = clean_json.split("</tool_call>")[0].strip()

        # 1. Parsing direto de JSON válido
        try:
            obj = json.loads(clean_json)
            if isinstance(obj, dict) and "name" in obj:
                parsed.append(obj)
                continue
            else:
                errors.append(f"JSON não é dict ou sem 'name': {clean_json[:50]}")
                continue
        except Exception:
            pass

        # 2. Fallback de fechamento de chaves
        recovered = False
        for pad in ["}", "}}", '"}}', '"}']:
            try:
                obj = json.loads(clean_json + pad)
                if isinstance(obj, dict) and "name" in obj:
                    parsed.append(obj)
                    recovered = True
                    break
            except Exception:
                pass
        if recovered:
            continue

        # 3. Fallback via regex para name e code
        name_m = re.search(r'"name"\s*:\s*"([^"]+)"', clean_json)
        if name_m:
            fn_name = name_m.group(1)
            code_m = re.search(r'"code"\s*:\s*"(.*)', clean_json, re.DOTALL)
            if code_m:
                raw_code = code_m.group(1)
                for end_pat in ['"}\n}', '"}', '"\n}']:
                    if end_pat in raw_code:
                        raw_code = raw_code.split(end_pat)[0]
                try:
                    raw_code = json.loads(f'"{raw_code}"')
                except Exception:
                    raw_code = raw_code.replace('\\n', '\n').replace('\\"', '"')
                parsed.append({"name": fn_name, "arguments": {"code": raw_code}})
            else:
                parsed.append({"name": fn_name, "arguments": {}})
        else:
            errors.append(f"JSON inválido em <tool_call>: {clean_json[:50]}")

    return parsed, has_xml_tag, errors


def sanitize_generated_code(code: str) -> str:
    """Higieniza código Python gerado pelo LLM para remover artefatos de fechamento de string/JSON."""
    if not code:
        return ""
    code = code.strip()
    # 1. Remover caudas de pseudo-código ou JS
    for noise in [".toString()", ".JSON", ".json()", ".enser"]:
        if noise in code:
            code = code.split(noise)[0]
    # 2. Remover pontuações/aspas residuais de fechamento de string
    while code and code[-1] in ['"', "'", '.', '}', ']', ';', '`']:
        code = code[:-1].strip()
    # 3. Balancear parênteses excedentes no final
    open_paren = code.count('(')
    close_paren = code.count(')')
    while close_paren > open_paren and code.endswith(')'):
        code = code[:-1].strip()
        close_paren -= 1
    return code


def test_python_code(code: str, expected_result: Any) -> Tuple[bool, str]:
    """Executa o código gerado em subprocesso isolado com timeout."""
    code = sanitize_generated_code(code)
    if not code or not code.strip():
        return False, "Código vazio"
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=3.0
        )
        stdout = res.stdout.strip()
        if expected_result is not None:
            exp_str = str(expected_result).strip()
            lines = [l.strip() for l in stdout.splitlines() if l.strip()]
            last_line = lines[-1] if lines else ""

            if stdout == exp_str or last_line == exp_str or exp_str in stdout:
                return True, f"Sucesso (saída={stdout})"
            try:
                if float(last_line) == float(exp_str):
                    return True, f"Sucesso numérico ({last_line} == {exp_str})"
            except (ValueError, TypeError):
                pass

            if res.returncode == 0:
                return False, f"Divergência: esperado '{exp_str}', obtido '{stdout}'"
            else:
                return False, f"Erro execução (código={res.returncode}): {res.stderr[:60]}"
        else:
            return (res.returncode == 0), f"Executado (código={res.returncode})"
    except subprocess.TimeoutExpired:
        return False, "Timeout (execução > 3.0s)"
    except Exception as e:
        return False, f"Exceção: {str(e)}"


def evaluate_single_sample(case: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    """Avalia individualmente uma amostra contra os critérios de Tool Calling."""
    expected_call = case.get("expected_call")
    should_call = (expected_call is not None)
    expected_result = case.get("expected_execution_result")

    tool_calls, has_xml, errors = extract_tool_calls_from_text(raw_output)

    format_ok = False
    selection_ok = False
    arguments_ok = False
    python_ok = False

    if not should_call:
        # Caso de Abstenção Conversacional
        format_ok = (not has_xml and len(tool_calls) == 0 and len(errors) == 0)
        selection_ok = (len(tool_calls) == 0)
        arguments_ok = True
        python_ok = True
        overall_pass = format_ok and selection_ok
    else:
        # Caso de Chamada de Ferramenta
        exp_name = expected_call.get("name")
        exp_args = expected_call.get("arguments", {})

        format_ok = (has_xml and len(tool_calls) > 0 and len(errors) == 0)
        
        # Encontrar match da tool
        matched_call = None
        for c in tool_calls:
            if c.get("name") == exp_name:
                selection_ok = True
                matched_call = c
                break

        # Argumentos
        if selection_ok and matched_call:
            c_args = matched_call.get("arguments", {})
            if isinstance(c_args, dict) and isinstance(exp_args, dict):
                # Todas as chaves requeridas no expected_call devem estar presentes
                if all(k in c_args for k in exp_args.keys()):
                    arguments_ok = True
            elif not exp_args:
                arguments_ok = True

        # Validação Python
        if exp_name == "execute_python":
            if matched_call and "code" in matched_call.get("arguments", {}):
                code = matched_call["arguments"]["code"]
                py_success, py_msg = test_python_code(code, expected_result)
                python_ok = py_success
            else:
                python_ok = False
        else:
            python_ok = True

        overall_pass = format_ok and selection_ok and arguments_ok and python_ok

    return {
        "format_ok": format_ok,
        "selection_ok": selection_ok,
        "arguments_ok": arguments_ok,
        "python_ok": python_ok,
        "overall_pass": overall_pass,
        "has_xml": has_xml,
        "calls_found": len(tool_calls),
        "errors": errors
    }


def run_battery_for_model(model_key: str, max_samples: int = None):
    meta = MODELS[model_key]
    model_name = meta["name"]
    model_path = meta["path"]

    print("\n" + "=" * 80)
    print(f"INICIANDO BATERIA DE 90 ITENS NO MODELO: {model_name}")
    print(f"Caminho do arquivo GGUF: {model_path}")
    print("=" * 80)

    if not os.path.exists(model_path):
        print(f"ERRO: Arquivo {model_path} não encontrado no disco!")
        return

    # 1. Carregar Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    # 2. Carregar Modelo llama-cpp
    t0_load = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False
    )
    print(f"Modelo carregado na GPU em {time.time() - t0_load:.2f} segundos!\n")

    # 3. Carregar Dataset Holdout
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    if max_samples:
        cases = cases[:max_samples]

    total_cases = len(cases)
    print(f"Total de amostras a avaliar: {total_cases} itens.\n")

    results = []
    cat_stats = defaultdict(lambda: {"total": 0, "pass": 0, "format": 0, "selection": 0, "args": 0, "python": 0})
    diff_stats = defaultdict(lambda: {"total": 0, "pass": 0})

    t_start = time.time()

    for idx, case in enumerate(cases, 1):
        c_id = case.get("id", f"case_{idx}")
        diff = case.get("difficulty", 1)
        cat = case.get("category", "unknown")
        tools = case.get("tools", [])
        tools_arg = tools if (isinstance(tools, list) and len(tools) > 0) else None

        # Montar prompt nativo com chat template
        prompt_str = tokenizer.apply_chat_template(
            case["messages"],
            tools=tools_arg,
            add_generation_prompt=True,
            tokenize=False
        )

        t_gen0 = time.time()
        res = llm(
            prompt_str,
            max_tokens=512,
            temperature=0.0,  # Avaliação determinística greedy
            repeat_penalty=1.03,
            stop=["<|im_end|>"]
        )
        gen_time = time.time() - t_gen0
        output_text = res["choices"][0]["text"].strip()
        n_tokens = res["usage"]["completion_tokens"]
        tps = n_tokens / max(1e-4, gen_time)

        eval_res = evaluate_single_sample(case, output_text)
        is_pass = eval_res["overall_pass"]

        # Atualizar acumuladores
        cat_stats[cat]["total"] += 1
        diff_stats[diff]["total"] += 1
        if is_pass:
            cat_stats[cat]["pass"] += 1
            diff_stats[diff]["pass"] += 1
        if eval_res["format_ok"]: cat_stats[cat]["format"] += 1
        if eval_res["selection_ok"]: cat_stats[cat]["selection"] += 1
        if eval_res["arguments_ok"]: cat_stats[cat]["args"] += 1
        if eval_res["python_ok"]: cat_stats[cat]["python"] += 1

        results.append({
            "idx": idx,
            "id": c_id,
            "difficulty": diff,
            "category": cat,
            "prompt_user": case["messages"][-1]["content"],
            "expected_call": case.get("expected_call"),
            "model_output": output_text,
            "eval": eval_res,
            "tokens": n_tokens,
            "gen_time": round(gen_time, 2),
            "tps": round(tps, 1)
        })

        # Log dinâmico a cada passo ou a cada 5 passos
        status_tag = "[PASS]" if is_pass else "[FAIL]"
        if idx % 5 == 0 or idx == total_cases:
            cur_passes = sum(1 for r in results if r["eval"]["overall_pass"])
            acc_so_far = (cur_passes / idx) * 100
            elapsed = time.time() - t_start
            eta_sec = (elapsed / idx) * (total_cases - idx)
            print(f"[{idx:02d}/{total_cases}] {status_tag} | Acurácia Atual: {acc_so_far:5.1f}% ({cur_passes}/{idx}) | TPS: {tps:4.1f} | ETA: {eta_sec/60:4.1f} min")
        elif not is_pass and diff == 3:
            # Notifica falhas em nível 3
            print(f"[{idx:02d}/{total_cases}] [FAIL L3] {c_id} ({cat}) - Saída: {output_text[:70]}...")

    # 4. Relatório Consolidado
    total_pass = sum(1 for r in results if r["eval"]["overall_pass"])
    overall_acc = (total_pass / total_cases) * 100
    total_time = (time.time() - t_start) / 60

    print("\n" + "=" * 80)
    print(f"RESULTADO FINAL DA BATERIA DE 90 ITENS: {model_name}")
    print(f"Tempo Total: {total_time:.2f} minutos | Acurácia Global: {overall_acc:.2f}% ({total_pass}/{total_cases})")
    print("=" * 80)

    print("\n--- DESEMPENHO POR NÍVEL DE DIFICULDADE ---")
    for d in sorted(diff_stats.keys()):
        st = diff_stats[d]
        pct = (st["pass"] / max(1, st["total"])) * 100
        print(f"Nível {d}: {pct:5.1f}% ({st['pass']}/{st['total']})")

    print("\n--- DESEMPENHO POR CATEGORIA ---")
    for cat, st in cat_stats.items():
        pct = (st["pass"] / max(1, st["total"])) * 100
        print(f"{cat:30s}: {pct:5.1f}% ({st['pass']}/{st['total']}) | Formato: {st['format']}/{st['total']} | Seleção: {st['selection']}/{st['total']}")

    # 5. Salvar Resultados em JSON
    out_file = os.path.join(ROOT_DIR, "data", f"eval_90_battery_{model_key}_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "model_key": model_key,
            "model_name": model_name,
            "model_path": model_path,
            "total_cases": total_cases,
            "total_pass": total_pass,
            "overall_accuracy_pct": round(overall_acc, 2),
            "total_time_min": round(total_time, 2),
            "difficulty_breakdown": dict(diff_stats),
            "category_breakdown": dict(cat_stats),
            "detailed_results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResultados detalhados gravados em: {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark de 90 itens nos modelos Q8")
    parser.add_argument("--model", type=str, choices=["tool_sft_q8", "tool_sft_ckpt150_q8", "checkpoints", "grpo_q8", "unrolled_sft", "base", "both", "all"], default="tool_sft_q8",
                        help="Modelo a avaliar: 'tool_sft_q8' (final 2 épocas), 'tool_sft_ckpt150_q8' (ckpt 150), 'checkpoints' (ambos), 'grpo_q8', 'unrolled_sft', 'base', 'both' ou 'all'")
    parser.add_argument("--max_samples", type=int, default=None, help="Limite de amostras para teste rápido")
    args = parser.parse_args()

    if args.model in ["tool_sft_q8", "checkpoints", "all"]:
        run_battery_for_model("tool_sft_q8", max_samples=args.max_samples)

    if args.model in ["tool_sft_ckpt150_q8", "checkpoints", "all"]:
        run_battery_for_model("tool_sft_ckpt150_q8", max_samples=args.max_samples)

    if args.model in ["grpo_q8", "all"]:
        run_battery_for_model("grpo_q8", max_samples=args.max_samples)

    if args.model in ["unrolled_sft", "both", "all"]:
        run_battery_for_model("unrolled_sft", max_samples=args.max_samples)

    if args.model in ["base", "both", "all"]:
        run_battery_for_model("base", max_samples=args.max_samples)


if __name__ == "__main__":
    main()

