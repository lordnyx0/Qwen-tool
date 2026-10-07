"""cleanup_batch2.py — lote mecanico 2 (decisao 2026-10-07): itens 2, 5, 6.

- Exclusao (12 uuids): 3 outliers token_length>5000 + 8 closers content-free
  (greetings) + 1 args com mojibake irrecuperavel. Mantidos: echores
  ("Confirmed — N"), terse de lote (mcp_tr_*) e respostas diretas (t_notool math).
- Backfill de properties: para tool DECLARADA com properties vazio mas CHAMADA
  com args, preenche properties a partir das chaves/tipos observados nas calls
  da propria amostra. Nao toca required, descriptions nem decoys nao-chamadas.
  Por construcao args ⊆ properties -> zero novas mascaras C1b.
- Retokeniza SÓ as linhas tocadas; atualiza token_len nos manifestos + stats.

Uso: python scripts/pipeline/cleanup_batch2.py --dry-run
       python scripts/pipeline/cleanup_batch2.py --apply   (backup .bakN antes)
"""
import argparse
import json
import os
import shutil
import sys
from collections import Counter
from glob import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PUR = os.path.join(ROOT, "data", "cleaned", "curriculum_3000_tool_sft_PURIFIED.jsonl")
MANIFESTS = [os.path.join(ROOT, "data", "split", "manifest.jsonl"),
             os.path.join(ROOT, "data", "split", "tracks", "generic", "manifest.jsonl"),
             os.path.join(ROOT, "data", "split", "tracks", "narrow", "manifest.jsonl")]
VIEWS = glob(os.path.join(ROOT, "data", "split", "views", "*.jsonl"))
BAKDIR = os.path.join(ROOT, "data", "cleaned", "backup")

EXCLUDE = {"purified_1715", "purified_1821", "s8d_016",
           "t_nav4_0027", "t_sml_0436", "t_sml_0629", "t_notool_0003",
           "t_notool_0005", "t_notool_0009", "t_notool_0010", "t_nav4_0032",
           "purified_0075"}


def jtype(v):
    if isinstance(v, bool):
        return "boolean"
    if isinstance(v, int):
        return "integer"
    if isinstance(v, float):
        return "number"
    if isinstance(v, list):
        return "array"
    if isinstance(v, dict):
        return "object"
    return "string"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    assert a.dry_run != a.apply, "use --dry-run OU --apply"

    rows = [json.loads(l) for l in open(PUR, encoding="utf-8") if l.strip()]
    have = {r["uuid"] for r in rows}
    missing = EXCLUDE - have
    assert not missing, f"exclusao com uuid ausente: {missing}"

    kept = [r for r in rows if r["uuid"] not in EXCLUDE]
    touched = set()
    n_backfill_samples = 0
    n_backfill_tools = 0
    for r in kept:
        observed = {}
        for m in r.get("messages", []):
            for tc in (m.get("tool_calls") or []):
                try:
                    aargs = json.loads((tc.get("function") or {}).get("arguments", "{}"))
                except Exception:
                    continue
                if isinstance(aargs, dict):
                    observed.setdefault((tc.get("function") or {}).get("name"), {}).update(aargs)
        for t in r.get("tools", []):
            fn = t.get("function", {})
            props = ((fn.get("parameters")) or {}).get("properties") or {}
            if len(props) == 0 and fn.get("name") in observed and observed[fn["name"]]:
                new_props = {k: {"type": jtype(v)} for k, v in observed[fn["name"]].items()}
                fn.setdefault("parameters", {}).setdefault("type", "object")
                fn["parameters"]["properties"] = new_props
                touched.add(r["uuid"])
                n_backfill_tools += 1
        if r["uuid"] in touched:
            n_backfill_samples += 1
    print(f"N={len(rows)} exclui={len(EXCLUDE)} backfill_amostras={n_backfill_samples} "
          f"backfill_tools={n_backfill_tools} retokeniza={len(touched)}")

    if a.dry_run:
        print("dry-run: nada escrito.")
        return

    os.makedirs(BAKDIR, exist_ok=True)

    def nums(d):
        out = []
        for fp in glob(os.path.join(d, os.path.basename(PUR) + ".bak*")):
            suf = fp.rsplit(".bak", 1)[-1]
            if suf.isdigit():
                out.append(int(suf))
        return out
    n = max(nums(os.path.dirname(PUR)) + nums(BAKDIR) + [0]) + 1
    bak = os.path.join(BAKDIR, os.path.basename(PUR) + f".bak{n}")
    shutil.copyfile(PUR, bak)
    print("backup:", bak)

    if touched:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(
            os.path.join(ROOT, "checkpoints", "qwen3loop_tool_sft_final"), local_files_only=True)
        by = {r["uuid"]: r for r in kept}
        for u in touched:
            r = by[u]
            r["token_length"] = len(tok.apply_chat_template(
                r["messages"], tools=r["tools"] or None, tokenize=True, return_dict=False))

    with open(PUR, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    keep = {r["uuid"] for r in kept}
    newlen = {r["uuid"]: r.get("token_length") for r in kept if r["uuid"] in touched}
    for mp in MANIFESTS + VIEWS:
        if not os.path.isfile(mp):
            continue
        lines = [json.loads(l) for l in open(mp, encoding="utf-8") if l.strip()]
        before = len(lines)
        with open(mp, "w", encoding="utf-8") as f:
            for e in lines:
                if e.get("uuid") in keep:
                    if e.get("uuid") in newlen and "token_len" in e:
                        e["token_len"] = newlen[e["uuid"]]
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"filtrado {os.path.relpath(mp, ROOT)}: {before} -> "
              f"{sum(1 for e in lines if e.get('uuid') in keep)}")

    stats = {"n": len(kept), "pillars": dict(Counter(s.get("pillar", "?") for s in kept)),
             "retr": sum(1 for s in kept if s.get("is_retrajectory"))}
    ts = sorted(s.get("token_length", 0) for s in kept)
    stats.update({"tok_med": ts[len(ts) // 2],
                  "b_short": sum(1 for s in kept if s.get("token_length", 0) <= 800),
                  "b_med": sum(1 for s in kept if 800 < s.get("token_length", 0) <= 2000),
                  "b_long": sum(1 for s in kept if s.get("token_length", 0) > 2000)})
    with open(os.path.join(ROOT, "data", "cleaned", "curriculum_purified_stats.json"),
              "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
