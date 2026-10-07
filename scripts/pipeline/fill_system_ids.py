"""fill_system_ids.py — lote mecanico: system ausente + ids ausentes (decisao 2026-10-07).

- System: prepend do system modal do pilar (computado dos que tem; fallback "?").
  Nao e sintese de observation: system e string fixa de condicionamento.
- Ids: mantem existentes; preenche ausentes como call_00i (largura dominante do
  arquivo); espelha tool_call_id por pareamento posicional (nunca sobrescreve).
- Retokeniza SÓ as linhas tocadas (tokenizer oficial) e atualiza token_len nos
  manifestos + stats. Ordem preservada.

Uso: python scripts/pipeline/fill_system_ids.py --dry-run
       python scripts/pipeline/fill_system_ids.py --apply   (backup .bakN antes)
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
BAKDIR = os.path.join(ROOT, "data", "cleaned", "backup")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    assert a.dry_run != a.apply, "use --dry-run OU --apply"

    rows = [json.loads(l) for l in open(PUR, encoding="utf-8") if l.strip()]

    widths = Counter()
    for r in rows:
        for m in r.get("messages", []):
            for tc in (m.get("tool_calls") or []):
                mm = re.fullmatch(r"call_(\d+)", tc.get("id") or "")
                if mm:
                    widths[len(mm.group(1))] += 1
    width = widths.most_common(1)[0][0] if widths else 3
    print("largura_id_dominante:", width, dict(widths))

    mode = {}
    for r in rows:
        ms = r.get("messages", [])
        if ms and ms[0].get("role") == "system":
            mode.setdefault(r.get("pillar", "?"), Counter())[ms[0].get("content", "")] += 1
    canon = {p: c.most_common(1)[0][0] for p, c in mode.items()}
    fallback = canon.get("?", "You are a helpful AI assistant equipped with specialized tools.")
    for p, c in mode.items():
        top, n = c.most_common(1)[0]
        print(f"system_modal pilar={p} share={n}/{sum(c.values())} len={len(top)}")

    touched, n_sys, n_tcid, n_tcid_left = set(), 0, 0, 0
    for r in rows:
        ms = r.get("messages", [])
        if ms and ms[0].get("role") != "system":
            ms.insert(0, {"role": "system", "content": canon.get(r.get("pillar", "?"), fallback)})
            touched.add(r["uuid"])
            n_sys += 1
        pending = []
        for m in ms:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                used = {tc.get("id") for tc in m["tool_calls"] if tc.get("id")}
                pending = []
                for i, tc in enumerate(m["tool_calls"], 1):
                    if not tc.get("id"):
                        cand = f"call_{i:0{width}d}"
                        k = i
                        while cand in used:
                            k += 1
                            cand = f"call_{k:0{width}d}"
                        tc["id"] = cand
                        used.add(cand)
                        touched.add(r["uuid"])
                        n_tcid += 1
                    pending.append(tc["id"])
            elif m.get("role") == "tool":
                if pending and not m.get("tool_call_id"):
                    m["tool_call_id"] = pending.pop(0)
                    touched.add(r["uuid"])
                    n_tcid += 1
                elif not m.get("tool_call_id") and not pending:
                    n_tcid_left += 1
            elif m.get("role") in ("assistant", "user"):
                if not (m.get("role") == "assistant" and m.get("tool_calls")):
                    pending = []
    print(f"systems_preenchidos={n_sys} ids_preenchidos={n_tcid} tcid_sem_par={n_tcid_left} linhas_tocadas={len(touched)}")

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
        by = {r["uuid"]: r for r in rows}
        for u in touched:
            r = by[u]
            r["token_length"] = len(tok.apply_chat_template(
                r["messages"], tools=r["tools"] or None, tokenize=True, return_dict=False))

    with open(PUR, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    newlen = {r["uuid"]: r.get("token_length") for r in rows if r["uuid"] in touched}
    for mp in MANIFESTS:
        if not os.path.isfile(mp):
            continue
        lines = [json.loads(l) for l in open(mp, encoding="utf-8") if l.strip()]
        upd = 0
        with open(mp, "w", encoding="utf-8") as f:
            for e in lines:
                if e.get("uuid") in newlen:
                    e["token_len"] = newlen[e["uuid"]]
                    upd += 1
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"token_len atualizado {os.path.relpath(mp, ROOT)}: {upd}")

    stats = {"n": len(rows), "pillars": dict(Counter(s.get("pillar", "?") for s in rows)),
             "retr": sum(1 for s in rows if s.get("is_retrajectory"))}
    ts = sorted(s.get("token_length", 0) for s in rows)
    stats.update({"tok_med": ts[len(ts) // 2],
                  "b_short": sum(1 for s in rows if s.get("token_length", 0) <= 800),
                  "b_med": sum(1 for s in rows if 800 < s.get("token_length", 0) <= 2000),
                  "b_long": sum(1 for s in rows if s.get("token_length", 0) > 2000)})
    with open(os.path.join(ROOT, "data", "cleaned", "curriculum_purified_stats.json"),
              "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
