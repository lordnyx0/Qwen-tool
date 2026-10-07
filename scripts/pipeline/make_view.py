"""make_view.py — materializa views do PURIFIED a partir do manifesto.

Uso: python scripts/pipeline/make_view.py --executor generic --calls multi
       --max-len 500 --out /tmp/view.jsonl [--min-tools 1 --source manual_s5]
Filtros: --executor {narrow,generic,mixed,abstention} (repetivel),
--calls single (<=1, exclui abstencao) | multi (>=2 sem retra) | retry (recuperacao de erro) | N exato,
--max-len / --min-len (token_len), --min-tools, --source (repetivel).
Saida: jsonl pronto p/ Kaggle + resumo impresso.
"""
import argparse
import json
import os
import sys

MANIFEST = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "..", "data", "split", "manifest.jsonl")
PURI = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "..", "data", "cleaned",
                    "curriculum_3000_tool_sft_PURIFIED.jsonl")


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "Programa"))
from app.core.dataset_loader import classify_effect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--executor", action="append", default=[])
    ap.add_argument("--calls", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--min-len", type=int, default=None)
    ap.add_argument("--min-tools", type=int, default=None)
    ap.add_argument("--source", action="append", default=[])
    ap.add_argument("--pillar", default=None)
    ap.add_argument("--exclude-pillar", action="append", default=[])
    ap.add_argument("--include-uuids", default=None,
                    help="JSON com {uuids:[...]} ou {no_tools:[...],with_catalog:[...]}: restringe a estes UUIDs")
    ap.add_argument("--subpillar", default=None)
    ap.add_argument("--effect", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mani = {}
    for l in open(MANIFEST, encoding="utf-8"):
        if l.strip():
            e = json.loads(l)
            mani[e["uuid"]] = e

    only = None
    if args.include_uuids:
        raw = json.load(open(args.include_uuids, encoding="utf-8"))
        if isinstance(raw, dict):
            only = set(raw.get("uuids", []) or raw.get("no_tools", []) + raw.get("with_catalog", []))
        elif isinstance(raw, list):
            only = set(x["uuid"] if isinstance(x, dict) else x for x in raw)
        else:
            only = set()
        print(f"include-uuids: {len(only)} alvos")

    def keep(m):
        if only is not None and m["uuid"] not in only:
            return False
        if args.executor and m["executor"] not in args.executor:
            return False
        if args.calls == "single" and not (m["n_calls"] == 1):
            return False
        if args.calls == "multi" and not (m.get("pattern", "multi") == "multi" and m["n_calls"] >= 2):
            return False
        if args.calls == "retry" and not (m.get("pattern") == "retry"):
            return False
        if args.calls and args.calls not in ("single", "multi", "retry"):
            if m["n_calls"] != int(args.calls):
                return False
        if args.max_len is not None and not (0 <= m["token_len"] <= args.max_len):
            return False
        if args.min_len is not None and not (m["token_len"] >= args.min_len):
            return False
        if args.min_tools is not None and not (m["n_tools"] >= args.min_tools):
            return False
        if args.source and m["source"] not in args.source:
            return False
        return True

    n = 0
    with open(PURI, encoding="utf-8") as fin, open(args.out, "w",
                                                  encoding="utf-8") as fout:
        for l in fin:
            if not l.strip():
                continue
            e = json.loads(l)
            if args.pillar and e.get("pillar") != args.pillar:
                continue
            if args.exclude_pillar and e.get("pillar") in args.exclude_pillar:
                continue
            if args.subpillar and e.get("subpillar") != args.subpillar:
                continue
            if args.effect and classify_effect(e) != args.effect:
                continue
            m = mani.get(e["uuid"])
            if m and keep(m):
                fout.write(l if l.endswith("\n") else l + "\n")
                n += 1
    print(f"view: {n} amostras -> {args.out}")


if __name__ == "__main__":
    main()
