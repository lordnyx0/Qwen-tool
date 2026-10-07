"""dedup_purified.py — dedup estratificado do PURIFIED (decisao 2026-10-07).

Chave: primeiro user normalizado + sequencia (tool_name, arguments canonicos)
(igual a scratch/audits/check_duplicates.py, modo padrao).
Keep-rule por grupo (deterministica): retra primeiro, depois menor catalogo,
depois uuid. Nao modifica linhas, so remove; ordem do arquivo preservada.
Filtra manifestos (global + tracks) e views para os uuids mantidos; regenera stats.

Uso: python scripts/pipeline/dedup_purified.py --dry-run
       python scripts/pipeline/dedup_purified.py --apply   (backup .bakN antes)
"""
import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from glob import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PUR = os.path.join(ROOT, "data", "cleaned", "curriculum_3000_tool_sft_PURIFIED.jsonl")
MANIFESTS = [os.path.join(ROOT, "data", "split", "manifest.jsonl"),
             os.path.join(ROOT, "data", "split", "tracks", "generic", "manifest.jsonl"),
             os.path.join(ROOT, "data", "split", "tracks", "narrow", "manifest.jsonl")]
VIEWS = glob(os.path.join(ROOT, "data", "split", "views", "*.jsonl"))
BAKDIR = os.path.join(ROOT, "data", "cleaned", "backup")


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def canon(a):
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return a.strip()
    return json.dumps(a, sort_keys=True, ensure_ascii=False)


def sig(r):
    u = next((norm(m.get("content")) for m in r.get("messages", []) if m.get("role") == "user"), "")
    t = tuple(((tc.get("function") or {}).get("name", ""), canon((tc.get("function") or {}).get("arguments")))
                for m in r.get("messages", []) for tc in (m.get("tool_calls") or []))
    return (u, t)


def keepkey(r):
    return (not r.get("is_retrajectory"), len(r.get("tools", [])), r.get("uuid", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    assert a.dry_run != a.apply, "use --dry-run OU --apply"

    rows = [json.loads(l) for l in open(PUR, encoding="utf-8") if l.strip()]
    groups = defaultdict(list)
    for r in rows:
        groups[sig(r)].append(r)
    keep = set()
    for g in groups.values():
        keep.add(sorted(g, key=keepkey)[0]["uuid"])
    drop = [r["uuid"] for r in rows if r["uuid"] not in keep]
    retr = sum(1 for r in rows if r["uuid"] in keep and r.get("is_retrajectory"))
    print(f"N={len(rows)} grupos_dup={sum(1 for g in groups.values() if len(g) > 1)} "
          f"remove={len(drop)} mantem={len(keep)} retra={retr} ({100 * retr / len(keep):.1f}%)")

    if a.dry_run:
        print("dry-run: nada escrito. Primeiros removiveis:", drop[:10])
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

    with open(PUR, "w", encoding="utf-8") as f:
        for r in rows:
            if r["uuid"] in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    for mp in MANIFESTS + VIEWS:
        if not os.path.isfile(mp):
            continue
        lines = [json.loads(l) for l in open(mp, encoding="utf-8") if l.strip()]
        before = len(lines)
        with open(mp, "w", encoding="utf-8") as f:
            for e in lines:
                if e.get("uuid") in keep:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"filtrado {os.path.relpath(mp, ROOT)}: {before} -> {sum(1 for e in lines if e.get('uuid') in keep)}")

    kept_rows = [r for r in rows if r["uuid"] in keep]
    stats = {"n": len(kept_rows), "pillars": dict(Counter(s.get("pillar", "?") for s in kept_rows)),
             "retr": sum(1 for s in kept_rows if s.get("is_retrajectory"))}
    ts = sorted(s.get("token_length", 0) for s in kept_rows)
    stats.update({"tok_med": ts[len(ts) // 2],
                  "b_short": sum(1 for s in kept_rows if s.get("token_length", 0) <= 800),
                  "b_med": sum(1 for s in kept_rows if 800 < s.get("token_length", 0) <= 2000),
                  "b_long": sum(1 for s in kept_rows if s.get("token_length", 0) > 2000)})
    with open(os.path.join(ROOT, "data", "cleaned", "curriculum_purified_stats.json"),
              "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
