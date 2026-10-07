"""apply_quarantine.py — remove uuids quarentenados do PURIFIED + manifest.

Uso: python scripts/pipeline/apply_quarantine.py <quarantine.jsonl> [--dry-run]
1) backup .bakN do PURIFIED e do manifest; 2) remove uuids dos dois arquivos;
3) retokenize no tokenizer oficial (token_length no PURIFIED, token_len no manifest);
4) stats. Aceita quarantine como JSON-array (qp10-qp14) ou jsonl.
"""
import json
import os
import shutil
import sys
from collections import Counter
from glob import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PUR = os.path.join(ROOT, "data", "cleaned", "curriculum_3000_tool_sft_PURIFIED.jsonl")
MANI = os.path.join(ROOT, "data", "split", "manifest.jsonl")
BAKDIR = os.path.join(ROOT, "data", "cleaned", "backup")
DRY = "--dry-run" in sys.argv
QFP = next(a for a in sys.argv[1:] if not a.startswith("--"))


def _bak(path, bakdir=None):
    d = bakdir or os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    nums = []
    for fp in glob(os.path.join(d, os.path.basename(path) + ".bak*")):
        suf = fp.rsplit(".bak", 1)[-1]
        if suf.isdigit():
            nums.append(int(suf))
    n = max(nums + [0]) + 1
    dst = os.path.join(d, os.path.basename(path) + f".bak{n}")
    if not DRY:
        shutil.copyfile(path, dst)
    print(("dry-run " if DRY else "") + "backup:", dst)


def _load_uuids(fp):
    txt = open(fp, encoding="utf-8").read().strip()
    if txt.startswith("["):
        return {r["uuid"] for r in json.loads(txt)}
    return {json.loads(l)["uuid"] for l in txt.splitlines() if l.strip()}


qids = _load_uuids(QFP)
print(f"quarentena {QFP}: {len(qids)} uuids")

rows = [json.loads(l) for l in open(PUR, encoding="utf-8") if l.strip()]
mani = [json.loads(l) for l in open(MANI, encoding="utf-8") if l.strip()]
hit_pur = sum(1 for r in rows if r["uuid"] in qids)
hit_mani = sum(1 for m in mani if m["uuid"] in qids)
print(f"atingidos: PURIFIED={hit_pur} manifest={hit_mani}")
assert hit_pur == len(qids), f"uuids ausentes no PURIFIED: {len(qids) - hit_pur}"
assert hit_mani == len(qids), f"uuids ausentes no manifest: {len(qids) - hit_mani}"
if DRY:
    print("dry-run OK — nada escrito.")
    sys.exit(0)

_bak(PUR, BAKDIR)
_bak(MANI)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    os.path.join(ROOT, "checkpoints", "qwen3loop_tool_sft_final"), local_files_only=True)


def _tlen(r):
    return len(tok.apply_chat_template(
        r["messages"], tools=r["tools"] or None, tokenize=True, return_dict=False))


rows = [r for r in rows if r["uuid"] not in qids]
for r in rows:
    r["token_length"] = _tlen(r)
by_uuid = {r["uuid"]: r for r in rows}
mani = [m for m in mani if m["uuid"] not in qids]
for m in mani:
    m["token_len"] = by_uuid[m["uuid"]]["token_length"]

with open(PUR, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(MANI, "w", encoding="utf-8") as f:
    for m in mani:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

stats = {"n": len(rows), "pillars": dict(Counter(s.get("pillar", "?") for s in rows)),
         "retr": sum(1 for s in rows if s.get("is_retrajectory"))}
ts = sorted(s["token_length"] for s in rows)
stats.update({"tok_med": ts[len(ts) // 2],
              "b_short": sum(1 for s in rows if s["token_length"] <= 800),
              "b_med": sum(1 for s in rows if 800 < s["token_length"] <= 2000),
              "b_long": sum(1 for s in rows if s["token_length"] > 2000)})
json.dump(stats, open(os.path.join(ROOT, "data", "cleaned", "curriculum_purified_stats.json"),
                      "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
