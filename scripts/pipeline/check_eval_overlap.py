"""
check_eval_overlap.py (V3 gate — anti-contaminacao)

Extrai assinaturas de instancias (numeros distintivos, literais quoted,
valores de args) de um staging e acusa matches contra as baterias congeladas
(benchmark EN 121, sonda 25, multiturn 12) + harness local P5 24
(`data/eval_nav_p5.jsonl`; fora das congeladas, com HARD gate proprio para
que tranches P4 nao treinem em cima dele). Excecao: scaffolding generico
(`print(`, `import math`, keywords) nunca e assinatura.

Uso:
    python scripts/pipeline/check_eval_overlap.py data/staging/t_trig2.jsonl
    python scripts/pipeline/check_eval_overlap.py --baseline   # T-lotes V2
"""

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EVAL_FILES = [
    os.path.join(ROOT, "data", "eval_benchmark_battery_v2_en.jsonl"),
    os.path.join(ROOT, "data", "eval_probe_canonical_25.jsonl"),
    os.path.join(ROOT, "data", "eval_multiturn_v1.jsonl"),
    os.path.join(ROOT, "data", "eval_nav_p5.jsonl"),
]

NUM = re.compile(r"\d{3,}(?:\.\d+)?|\d+\.\d+")
QUOTED = re.compile(r"'([^']{6,})'|\"([^\"]{6,})\"")

# Nomes de chaves/parametros: nunca sao instancia (evita 'command', 'path', ...)
KEYS = {"command", "path", "query", "code", "city", "unit", "symbol", "url",
        "message", "message_text", "subject", "body", "body_markdown", "title",
        "description", "status", "stdout", "result", "answer", "value",
        "expression", "content", "text", "name", "id", "type"}


def signatures_of_text(t):
    sigs = set()
    for m in NUM.finditer(t or ""):
        sigs.add(("num", m.group(0)))
    for m in QUOTED.finditer(t or ""):
        s = (m.group(1) or m.group(2)).strip()
        if len(s) >= 6 and s.lower() not in KEYS:
            sigs.add(("lit", s))
    return sigs


def item_text(it):
    parts = []
    for m in it.get("messages", []):
        parts.append(m.get("content") or "")
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {}) or {}
            a = fn.get("arguments", "")
            parts.append(a if isinstance(a, str) else json.dumps(a, ensure_ascii=False))
    ea = it.get("expected_args") or {}
    if isinstance(ea, str):
        try:
            ea = json.loads(ea)
        except Exception:
            ea = {}
    for v in (ea.values() if isinstance(ea, dict) else [ea]):
        parts.append(str(v))
    return "\n".join(parts)


def load_eval():
    sig2ids = {}
    for fp in EVAL_FILES:
        tag = os.path.basename(fp)
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            it = json.loads(line)
            for s in signatures_of_text(item_text(it)):
                sig2ids.setdefault(s, []).append(f"{tag}:{it.get('id')}")
    return sig2ids


def check_file(path, sig2ids):
    hard = []
    soft = []
    n = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        n += 1
        it = json.loads(line)
        iid = it.get("id", it.get("uuid", f"linha{n}"))
        user_text = "\n".join(m.get("content") or "" for m in it.get("messages", [])
                              if m.get("role") == "user")
        rest_text = item_text(it)
        user_sigs = signatures_of_text(user_text)
        for s in user_sigs:
            if s in sig2ids:
                hard.append((iid, s, sig2ids[s][:3]))
        rest_sigs = signatures_of_text(rest_text) - user_sigs
        for s in rest_sigs:
            if s in sig2ids:
                soft.append((iid, s, sig2ids[s][:3]))
    return n, hard, soft


def main():
    sig2ids = load_eval()
    print(f"assinaturas eval: {len(sig2ids)}")
    if "--baseline" in sys.argv and len(sys.argv) < 3:
        for fp in ["data/staging/t_sem.jsonl", "data/staging/t_abs.jsonl",
                   "data/staging/t_trig.jsonl", "data/staging/t_dec.jsonl"]:
            p = os.path.join(ROOT, fp)
            if not os.path.exists(p):
                continue
            n, hard, soft = check_file(p, sig2ids)
            print(f"{fp}: {n} amostras, {len(hard)} hard + {len(soft)} soft")
            seen = set()
            for iid, s, refs in hard + soft:
                if iid not in seen:
                    seen.add(iid)
                    print(f"  - {iid}: {s[0]}={s[1][:50]!r} em {refs[0]}")
        return
    if len(sys.argv) < 2:
        print("uso: check_eval_overlap.py <staging.jsonl> [--baseline]")
        sys.exit(2)
    n, hard, soft = check_file(sys.argv[1], sig2ids)
    print(f"{sys.argv[1]}: {n} amostras, {len(hard)} HARD + {len(soft)} soft")
    for iid, s, refs in hard:
        print(f"  HARD - {iid}: {s[0]}={s[1][:60]!r} em {refs}")
    for iid, s, refs in soft:
        print(f"  soft - {iid}: {s[0]}={s[1][:60]!r} em {refs}")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
