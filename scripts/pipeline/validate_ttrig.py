"""
validate_ttrig.py (V2 Etapa 3 — gate mecanico do lote T-TRIG)

Valida amostras trigger candidate (staging): pergunta quantitativa direta ->
`execute_python` com snippet `print`, stdout VERIFICADO por execucao.
Aceita dois esquemas: amostra treino (messages c/ tool_calls + tools) ou
item de bateria (expected_args.code + ground_truth_stdout).

Checa por item:
- ha exatamente 1 call `execute_python` com arg `code` nao-vazio;
- estilo snippet (rejeita heredoc `<<`, `open(`, `input(` — T-TRIG e print-direto);
- executa o codigo em sandbox; stdout == ground_truth_stdout (match flexivel);
- `retra:false` quando o campo existir (lote novo e sempre retra:false).

Uso:
    python scripts/pipeline/validate_ttrig.py data/staging/t_trig.jsonl
    python scripts/pipeline/validate_ttrig.py --selftest   # bateria L3 + negativos
"""

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "eval"))
from benchmark_battery_v2 import run_snippet

BANNED = ("<<", "open(", "input(", "os.system", "subprocess")


def flex_match(out: str, exp: str) -> bool:
    exp = str(exp).strip()
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    last = lines[-1] if lines else ""
    if out.strip() == exp or last == exp or exp in out.strip():
        return True
    try:
        return float(last) == float(exp)
    except (ValueError, TypeError):
        return False


def extract(item):
    """Retorna (code, ground_truth, retra_flag_ou_None, erros_estrutura)."""
    if "expected_args" in item:
        code = (item.get("expected_args") or {}).get("code", "")
        gt = item.get("ground_truth_stdout", "")
        return code, gt, None, []
    errs = []
    calls = [tc for m in item.get("messages", [])
             for tc in (m.get("tool_calls") or [])]
    py = [tc for tc in calls
          if (tc.get("function", {}) or {}).get("name") == "execute_python"]
    if len(py) != 1:
        errs.append(f"calls execute_python != 1 (achado {len(py)})")
        return "", "", item.get("is_retrajectory"), errs
    a = (py[0].get("function", {}) or {}).get("arguments", {})
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            errs.append("arguments nao-JSON")
            return "", "", item.get("is_retrajectory"), errs
    code = a.get("code", "") or ""
    gt = item.get("ground_truth_stdout", "")
    if not str(gt).strip():
        # schema de treino: stdout mora no observation da tool
        for m in item.get("messages", []):
            if m.get("role") != "tool":
                continue
            try:
                obs = json.loads(m.get("content", "") or "")
            except Exception:
                continue
            if isinstance(obs, dict):
                for k in ("stdout", "output", "result", "answer"):
                    if str(obs.get(k, "")).strip():
                        gt = obs[k]
                        break
            if str(gt).strip():
                break
    if item.get("is_retrajectory") not in (False, None):
        errs.append("lote T-TRIG exige retra:false")
    return code, gt, item.get("is_retrajectory"), errs


def check_item(item):
    code, gt, _, errs = extract(item)
    errs = list(errs)
    if not (code or "").strip():
        errs.append("code vazio")
        return False, errs
    if any(b in code for b in BANNED):
        errs.append("estilo arquivo/heredoc (T-TRIG e snippet print-direto)")
    if not str(gt).strip():
        errs.append("ground_truth_stdout vazio")
        return False, errs
    ok, out = run_snippet(code)
    if not ok:
        errs.append(f"codigo nao executa ({out[:80]})")
        return False, errs
    if not flex_match(out, gt):
        errs.append(f"stdout '{out[:80]}' != gt '{str(gt)[:80]}'")
        return False, errs
    return (len(errs) == 0), errs


def run_file(path):
    items = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    bad = 0
    for i, it in enumerate(items):
        ok, errs = check_item(it)
        if not ok:
            bad += 1
            print(f"[FAIL] {it.get('id', i)}: " + " | ".join(errs))
    print(f"{path}: {len(items)-bad}/{len(items)} OK")
    return bad


def selftest():
    print("== positivos: 30 L3 da bateria EN (referencias devem passar) ==")
    bat = [json.loads(l) for l in
           open(os.path.join(ROOT, "data", "eval_benchmark_battery_v2_en.jsonl"),
                encoding="utf-8") if l.strip()]
    l3 = [it for it in bat if it["category"] == "python_algorithmic_reasoning"]
    bad = sum(1 for it in l3 if not check_item(it)[0])
    print(f"L3: {len(l3)-bad}/{len(l3)} OK")
    print("== negativos sinteticos (devem reprovar) ==")
    negs = [
        {"id": "neg1", "expected_args": {"code": "print(1/0)"}, "ground_truth_stdout": "x"},
        {"id": "neg2", "expected_args": {"code": "print(2+2)"}, "ground_truth_stdout": "5"},
        {"id": "neg3", "expected_args": {"code": "cat > a.py << 'EOF'\nprint(1)\nEOF"}, "ground_truth_stdout": "1"},
        {"id": "neg4", "expected_args": {}, "ground_truth_stdout": "1"},
    ]
    caught = sum(1 for it in negs if not check_item(it)[0])
    print(f"negativos capturados: {caught}/{len(negs)}")
    return 0 if (bad == 0 and caught == len(negs)) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.path:
        ap.error("informe o staging ou --selftest")
    sys.exit(1 if run_file(args.path) else 0)


if __name__ == "__main__":
    main()
