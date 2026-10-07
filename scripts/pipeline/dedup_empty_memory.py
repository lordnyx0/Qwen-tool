"""
dedup_empty_memory.py (V2 Etapa 1 — mecanico)

Remove excedentes de duplicata EXATA em chamadas `archival_memory_search`
com query vazia (""/"{}"): chave = (primeiro user[:80], arguments serializado).
Mantem a primeira ocorrencia de cada grupo (retrieve_all unicos preservados).

Uso:
    python scripts/pipeline/dedup_empty_memory.py --dry-run
    python scripts/pipeline/dedup_empty_memory.py --apply   # backup .bakN + valida
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PUR = os.path.join(ROOT, "data", "cleaned", "curriculum_3000_tool_sft_PURIFIED.jsonl")


def load_args(a):
    if isinstance(a, str):
        try:
            return json.loads(a)
        except Exception:
            return {}
    return a if isinstance(a, dict) else {}


def find_targets(items):
    seen = {}
    drop = []
    groups = Counter()
    for idx, it in enumerate(items):
        user = next((m.get("content", "") or "" for m in it.get("messages", [])
                     if m.get("role") == "user"), "")
        for m in it.get("messages", []):
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function", {})
                if fn.get("name") != "archival_memory_search":
                    continue
                a = load_args(fn.get("arguments", {}))
                if (a.get("query", None) or "") not in ("", "{}"):
                    continue
                key = (user[:80], json.dumps(a, sort_keys=True))
                groups[key] += 1
                if key in seen:
                    drop.append(idx)
                    break
                seen[key] = idx
    return drop, sum(1 for v in groups.values() if v > 1)


def next_backup():
    n = 1
    while os.path.exists(f"{PUR}.bak{n}"):
        n += 1
    return f"{PUR}.bak{n}"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(PUR, encoding="utf-8") if l.strip()]
    drop, ngroups = find_targets(items)
    ids = [items[i].get("id", f"linha{i}") for i in drop]
    print(f"N atual: {len(items)} | grupos dup: {ngroups} | para remover: {len(drop)}")
    for i in ids:
        print("  -", i)

    if args.dry_run:
        print("dry-run: nada alterado.")
        return

    bak = next_backup()
    with open(PUR, encoding="utf-8") as fsrc, open(bak, "w", encoding="utf-8") as fdst:
        fdst.write(fsrc.read())
    print("backup:", bak)
    keep = [it for i, it in enumerate(items) if i not in set(drop)]
    with open(PUR, "w", encoding="utf-8") as f:
        for it in keep:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"aplicado: N {len(items)} -> {len(keep)}")


if __name__ == "__main__":
    sys.exit(main())
