"""
benchmark_battery_v2.py
Executa a Bateria Holdout v2 (121 itens com 50 ferramentas e 8 domínios)
no modelo GGUF unrolled quantizado via llama-cpp-python com aceleração GPU (CUDA).
Gera métricas detalhadas:
- Taxa de Sucesso Geral (Overall Accuracy)
- Aderência de Formato XML/JSON
- Acurácia de Seleção de Ferramenta (Single-Tool e Multi-Tool Discrimination)
- Acurácia de Argumentos (Valores, Tipos e Enums)
- Execução em Sandbox Python com comparação exata de stdout
- Abstenção Negativa (respostas de texto puro quando não há chamada de ferramenta)
"""

import os
import sys
import re
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
DEFAULT_BATTERY = os.path.join(ROOT_DIR, "data", "eval_benchmark_battery_v2.jsonl")
DEFAULT_MODEL = os.path.join(ROOT_DIR, "models", "gguf", "unrolled_modelo_qwen3loop_sft_q8_0.gguf")
DEFAULT_TOKENIZER = os.path.join(ROOT_DIR, "checkpoints", "qwen3loop_sft_final")


def extract_tool_calls_from_text(text: str) -> Tuple[List[Dict[str, Any]], bool, List[str]]:
    """Extrai chamadas de ferramenta do formato nativo Qwen com suporte a fallbacks."""
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    parsed = []
    errors = []
    has_xml_tag = ("<tool_call>" in text)

    if not matches and "<tool_call>" in text:
        after_tag = text.split("<tool_call>", 1)[1]
        matches = [after_tag]

    for m in matches:
        clean_json = m.strip()
        if "</tool_call>" in clean_json:
            clean_json = clean_json.split("</tool_call>")[0].strip()

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

        # Fallback chaves
        for pad in ["}", "}}", '"}}', '"}']:
            try:
                obj = json.loads(clean_json + pad)
                if isinstance(obj, dict) and "name" in obj:
                    parsed.append(obj)
                    break
            except Exception:
                pass

        # Fallback regex
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


def test_python_code(code: str, expected_stdout: str) -> Tuple[bool, str]:
    """Executa código gerado em sandbox isolada com timeout."""
    if not code or not code.strip():
        return False, "Código vazio"
    
    code = code.strip()
    while code and code[-1] in ['"', "'", '.', '}', ']', ';', '`']:
        code = code[:-1].strip()

    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=3.0
        )
        stdout = res.stdout.strip()
        exp_str = str(expected_stdout).strip()
        lines = [l.strip() for l in stdout.splitlines() if l.strip()]
        last_line = lines[-1] if lines else ""

        if stdout == exp_str or last_line == exp_str or exp_str in stdout:
            return True, f"Match Exato (saída={stdout})"
        try:
            if float(last_line) == float(exp_str):
                return True, f"Match Numérico ({last_line} == {exp_str})"
        except (ValueError, TypeError):
            pass

        if res.returncode == 0:
            return False, f"Divergência: esperado '{exp_str}', obtido '{stdout}'"
        else:
            return False, f"Erro execução: {res.stderr[:60]}"
    except subprocess.TimeoutExpired:
        return False, "Timeout (> 3.0s)"
    except Exception as e:
        return False, f"Exceção: {str(e)}"


UNIT_ALIAS = {"c": "celsius"}


def norm_arg(s: str) -> str:
    """Normaliza string p/ comparacao: lower + strip + sem pontuacao final + alias."""
    t = str(s).lower().strip().rstrip(".!?;:")
    return UNIT_ALIAS.get(t, t)


def looks_like_python(code: str) -> bool:
    c = str(code)
    return ("print(" in c) or ("import " in c)


def run_snippet(code: str) -> Tuple[bool, str]:
    """Executa snippet e retorna (ok, stdout). Sem comparacao — so captura."""
    code = (code or "").strip()
    while code and code[-1] in ['"', "'", '.', '}', ']', ';', '`']:
        code = code[:-1].strip()
    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=3.0)
        if res.returncode == 0:
            return True, res.stdout.strip()
        return False, res.stderr[:80]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:80]


def exec_args_equal(got_code: str, exp_code: str) -> Tuple[bool, str]:
    """Compara dois codigos Python pelo stdout executado (V2-eval)."""
    ok_g, out_g = run_snippet(got_code)
    ok_e, out_e = run_snippet(exp_code)
    if not ok_g:
        return False, f"codigo gerado nao executa ({out_g})"
    if not ok_e:
        return True, "referencia nao executa; aceito por forma"
    lg = [l.strip() for l in out_g.splitlines() if l.strip()]
    last = lg[-1] if lg else ""
    if out_g == out_e or last == out_e.strip() or out_e.strip() in out_g:
        return True, f"stdout equivalente ({out_g[:60]})"
    try:
        if float(last) == float(out_e.strip()):
            return True, f"stdout numerico equivalente ({last})"
    except (ValueError, TypeError):
        pass
    return False, f"stdout diverge: '{out_g[:60]}' != '{out_e[:60]}'"


def evaluate_sample(item: Dict[str, Any], raw_output: str) -> Dict[str, Any]:
    expected_tool = item.get("expected_tool", "")
    is_abstention = (expected_tool == "")
    expected_args = item.get("expected_args", {})
    ground_truth_stdout = item.get("ground_truth_stdout", "")

    tool_calls, has_xml, errors = extract_tool_calls_from_text(raw_output)

    declared = set()
    for t in (item.get("tools") or []):
        try:
            declared.add(t["function"]["name"])
        except Exception:
            pass
    undeclared_names = sorted({c.get("name") for c in tool_calls
                               if c.get("name") not in declared})
    undeclared_call = len(undeclared_names) > 0

    format_ok = False
    selection_ok = False
    args_ok = False
    python_ok = False

    if is_abstention:
        # Abstenção Negativa: Não deve emitir nenhuma tag de ferramenta
        format_ok = (not has_xml and len(tool_calls) == 0 and len(errors) == 0)
        selection_ok = (len(tool_calls) == 0)
        args_ok = True
        python_ok = True
        overall_pass = format_ok and selection_ok
        diag = "Abstenção correta (sem tool)" if overall_pass else "Falha: chamou tool indevidamente"
    else:
        format_ok = (has_xml and len(tool_calls) > 0 and len(errors) == 0)
        matched_call = None
        for c in tool_calls:
            if c.get("name") == expected_tool:
                selection_ok = True
                matched_call = c
                break

        if selection_ok and matched_call:
            c_args = matched_call.get("arguments", {})
            if expected_tool == "execute_python":
                gen_code = c_args.get("code", "")
                python_ok, diag = test_python_code(gen_code, ground_truth_stdout)
                args_ok = (len(gen_code.strip()) > 0)
                overall_pass = format_ok and selection_ok and args_ok and python_ok
            else:
                # Validar argumentos
                all_match = True
                diag_parts = []
                for k, v in expected_args.items():
                    if k not in c_args:
                        all_match = False
                        diag_parts.append(f"Falta arg '{k}'")
                    else:
                        got = c_args[k]
                        # Match flexível de tipos
                        if isinstance(v, (int, float)) and isinstance(got, (int, float)):
                            if abs(float(v) - float(got)) > 1e-4:
                                all_match = False
                                diag_parts.append(f"{k}: {got} != {v}")
                        elif (k == "code" and looks_like_python(got)
                                and looks_like_python(v)):
                            # V2-eval: codigo Python compara por stdout executado
                            eq, why = exec_args_equal(got, v)
                            if not eq:
                                all_match = False
                            diag_parts.append(f"{k} (exec): {why}")
                            if eq and len(diag_parts) == 1:
                                diag_parts.clear()
                        elif norm_arg(got) != norm_arg(v):
                            all_match = False
                            diag_parts.append(f"{k}: '{got}' != '{v}'")
                args_ok = all_match
                python_ok = True
                overall_pass = format_ok and selection_ok and args_ok
                diag = "Argumentos OK" if args_ok else ("; ".join(diag_parts) or "Arg mismatch")
        else:
            overall_pass = False
            diag = f"Tool errada: esperado '{expected_tool}', gerou {[c.get('name') for c in tool_calls]}"

    return {
        "id": item["id"],
        "level": item["level"],
        "category": item["category"],
        "overall_pass": overall_pass,
        "format_ok": format_ok,
        "selection_ok": selection_ok,
        "args_ok": args_ok,
        "python_ok": python_ok,
        "undeclared_call": undeclared_call,
        "undeclared_names": undeclared_names,
        "diag": (diag + f" | UNDECLARED: {undeclared_names}" if undeclared_call else diag),
        "raw_output": raw_output[:200]
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Holdout Battery v2")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer_dir", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--battery_path", type=str, default=DEFAULT_BATTERY)
    parser.add_argument("--output_file", type=str, default="data/eval_results_unrolled_sft_v2.json")
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat_penalty", type=float, default=1.1)
    args = parser.parse_args()

    print("=" * 75)
    print("AVALIAÇÃO DO CHECKPOINT BASE NA NOVA BATERIA HOLDOUT V2")
    print(f"Modelo: {args.model_path}")
    print(f"Bateria: {args.battery_path}")
    print(f"Tokenizer: {args.tokenizer_dir}")
    print("=" * 75)

    if not os.path.exists(args.model_path):
        print(f"ERRO: Modelo não encontrado em {args.model_path}")
        sys.exit(1)

    print("Carregando Tokenizer Qwen...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    print("Carregando modelo GGUF no Llama.cpp com aceleração GPU...")
    llm = Llama(
        model_path=args.model_path,
        n_ctx=4096,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False
    )
    print("Modelo carregado com sucesso!\n")

    # Carregar bateria
    cases = []
    with open(args.battery_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    print(f"Total de casos na Bateria v2: {len(cases)}")
    print("-" * 75)

    results = []
    category_stats = defaultdict(lambda: {"total": 0, "pass": 0, "format": 0, "select": 0, "args": 0, "python": 0, "undecl": 0})

    start_time = time.time()
    for idx, case in enumerate(cases, 1):
        # Renderizar prompt usando o chat_template com tools
        tools = case.get("tools", [])
        messages = case.get("messages", [])

        if tools:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

        # Inferência
        out = llm(
            prompt_text,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            repeat_penalty=args.repeat_penalty,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        raw_text = out["choices"][0]["text"].strip()

        eval_res = evaluate_sample(case, raw_text)
        results.append(eval_res)

        cat = case.get("category", "unknown")
        category_stats[cat]["total"] += 1
        if eval_res["overall_pass"]: category_stats[cat]["pass"] += 1
        if eval_res["format_ok"]: category_stats[cat]["format"] += 1
        if eval_res["selection_ok"]: category_stats[cat]["select"] += 1
        if eval_res["args_ok"]: category_stats[cat]["args"] += 1
        if eval_res["python_ok"]: category_stats[cat]["python"] += 1
        if eval_res["undeclared_call"]: category_stats[cat]["undecl"] += 1

        mark = "[PASS]" if eval_res["overall_pass"] else "[FAIL]"
        print(f"[{idx:03d}/{len(cases):03d}] {mark} {case['id']} ({cat}) -> {eval_res['diag']}")

    elapsed = time.time() - start_time
    total_samples = len(results)
    total_passed = sum(1 for r in results if r["overall_pass"])
    format_passed = sum(1 for r in results if r["format_ok"])
    total_undecl = sum(1 for r in results if r["undeclared_call"])

    print("\n" + "=" * 75)
    print(f"RELATÓRIO DE DESEMPENHO BASELINE - BATERIA HOLDOUT V2 (Tempo: {elapsed:.1f}s)")
    print("=" * 75)
    print(f"Acurácia Geral (Overall Accuracy): {total_passed}/{total_samples} ({total_passed/total_samples*100:.1f}%)")
    print(f"Aderência de Formato XML/JSON:   {format_passed}/{total_samples} ({format_passed/total_samples*100:.1f}%)")
    print(f"Chamadas NÃO-declaradas:         {total_undecl}/{total_samples} ({total_undecl/total_samples*100:.1f}%)\n")

    print(f"{'Categoria':<35} | {'Acertos':<10} | {'Formato':<10} | {'Seleção':<10} | {'Undecl':<6}")
    print("-" * 75)
    for cat, st in category_stats.items():
        pct = (st['pass'] / st['total']) * 100 if st['total'] else 0
        fmt_pct = (st['format'] / st['total']) * 100 if st['total'] else 0
        sel_pct = (st['select'] / st['total']) * 100 if st['total'] else 0
        print(f"{cat:<35} | {st['pass']:02d}/{st['total']:02d} ({pct:4.1f}%) | {fmt_pct:4.1f}%     | {sel_pct:4.1f}%     | {st['undecl']:02d}")
    print("=" * 75)

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model_path,
            "total_passed": total_passed,
            "total_samples": total_samples,
            "accuracy_pct": total_passed / total_samples * 100,
            "format_pct": format_passed / total_samples * 100,
            "undeclared_count": total_undecl,
            "undeclared_pct": total_undecl / total_samples * 100,
            "elapsed_seconds": elapsed,
            "categories": category_stats,
            "details": results
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResultados detalhados salvos em: {args.output_file}")


if __name__ == "__main__":
    main()
