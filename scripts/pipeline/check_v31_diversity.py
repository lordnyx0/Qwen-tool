"""check_v31_diversity.py — gate anti-template para staging V3.1.
Uso: python scripts/pipeline/check_v31_diversity.py data/staging/v31/*.jsonl
Acusa: (1) openers de <think> repetidos (>3x); (2) frases de desistência/recusa
no turno final do assistant; (3) retry idêntico (mesma tool+args 2x seguidas).
Saída 0 = limpo; 1 = violações (lista todas).
"""
import json
import re
import sys
from collections import Counter

THINK_PAT = re.compile(r"<think>\s*(.*?)\s*(?:</think>|$)", re.S)
ERR_PAT = re.compile(
    r"FileNotFoundError|ENOENT|EISDIR|Traceback|SyntaxError|No such file"
    r"|not found|is a directory|permission denied|exit [1-9]",
    re.I,
)
GIVE_UP_PAT = re.compile(
    r"\bi\s*(can'?t|cannot|am\s+unable|'?m\s+unable|won'?t\s+be\s+able)\b"
    r"|\bnot\s+possible\b|\bgiving\s+up\b|\bunable\s+to\s+(proceed|continue|do\b)",
    re.I,
)

def think_openers(text):
    out = []
    for m in THINK_PAT.finditer(text or ""):
        first = (m.group(1).strip().split("\n") or [""])[0].strip().lower()
        if first:
            out.append(first)
    return out

def main(paths):
    opener_count = Counter()
    opener_where = {}
    violations = []
    n = 0
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                n += 1
                o = json.loads(line)
                uid = o.get("uuid", f"{p}:{ln}")
                msgs = o.get("messages", [])
                assts = [m for m in msgs if m.get("role") == "assistant"]
                # 1. openers
                for m in assts:
                    for op in think_openers(m.get("content", "")):
                        opener_count[op] += 1
                        opener_where.setdefault(op, []).append(uid)
                # 2. desistência no turno final
                if assts and not (assts[-1].get("tool_calls") or []):
                    if GIVE_UP_PAT.search(assts[-1].get("content", "")):
                        violations.append(f"GIVE_UP {uid}: turno final com desistência/recusa")
                # 3. retry idêntico APÓS ERRO (mesma tool+args depois de falha;
                # releitura após conteúdo parcial/ok é continuação legítima, não retry)
                prev_key = None
                last_tool_err = False
                for m in msgs:
                    if m.get("role") == "assistant":
                        for tc in (m.get("tool_calls") or []):
                            fn = ((tc.get("function") or {}).get("name"))
                            ag = ((tc.get("function") or {}).get("arguments"))
                            key = (fn, ag if isinstance(ag, str) else json.dumps(ag, sort_keys=True))
                            if key == prev_key and last_tool_err:
                                violations.append(f"RETRY {uid}: {fn} repetido com mesmos args após erro")
                            prev_key = key
                            last_tool_err = False
                    elif m.get("role") == "tool":
                        last_tool_err = bool(ERR_PAT.search(m.get("content", "")))
    for op, c in opener_count.items():
        if c > 3:
            violations.append(f"OPENER x{c}: {op!r} em {opener_where[op][:5]}")
    print(f"amostras: {n} | openers distintos: {len(opener_count)} | violações: {len(violations)}")
    for v in violations[:50]:
        print(" -", v)
    return 1 if violations else 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("uso: check_v31_diversity.py <staging.jsonl> [...]")
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
