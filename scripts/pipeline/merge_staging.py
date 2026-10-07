"""
merge_staging.py — pipeline canônico de merge V2 (ver 09_v2_roadmap.md).
Uso: python scripts/merge_staging.py <arquivo_staging.jsonl> [mais arquivos...]
1) backup .bakN do PURIFIED; 2) valida staging (mesmas regras do validate_sft);
3) append (SEM prune: tools declaradas-nao-chamadas sao sinal legitimo —
distratores, catalogos de irrelevancia); 4) retokenize no tokenizer oficial;
5) stats. O sync p/ Kaggle/kaggle_sft_data é passo manual separado (Copy-Item).
"""
import json
import os
import re
import shutil
import sys
from collections import Counter
from glob import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PUR = os.path.join(ROOT, "data", "cleaned", "curriculum_3000_tool_sft_PURIFIED.jsonl")
GENERIC = {"command completed successfully with exit code 0.", "execution succeeded with output.",
           "file content loaded successfully.", '{"status":"ok"}', '{"status": "ok"}'}

# backup numerado (cadeia em data/cleaned/ + data/cleaned/backup/)
BAKDIR = os.path.join(ROOT, "data", "cleaned", "backup")
os.makedirs(BAKDIR, exist_ok=True)
def _bak_nums(d):
    nums = []
    for fp in glob(os.path.join(d, os.path.basename(PUR) + ".bak*")):
        suf = fp.rsplit(".bak", 1)[-1]
        if suf.isdigit():
            nums.append(int(suf))
    return nums
n = max(_bak_nums(os.path.dirname(PUR)) + _bak_nums(BAKDIR) + [0]) + 1
bak_fp = os.path.join(BAKDIR, os.path.basename(PUR) + f".bak{n}")
shutil.copyfile(PUR, bak_fp)
print("backup:", bak_fp)

rows = [json.loads(l) for l in open(PUR, encoding="utf-8") if l.strip()]
have = set(r["uuid"] for r in rows)
added = 0
for fp in sys.argv[1:]:
    for l in open(fp, encoding="utf-8"):
        if not l.strip():
            continue
        r = json.loads(l)
        assert r["uuid"] not in have, "uuid duplicado: " + r["uuid"]
        assert r["messages"] and r["messages"][-1].get("role") == "assistant", r["uuid"]
        assert r["messages"][0].get("role") == "system", "sem system: " + r["uuid"]
        decl = set(t["function"]["name"] for t in r["tools"])
        for m in r["messages"]:
            for tc in (m.get("tool_calls") or []):
                assert tc["function"]["name"] in decl, r["uuid"]
                json.loads(tc["function"]["arguments"])
            if m.get("role") == "tool":
                c = m.get("content", "")
                assert c.strip() and c.strip() not in GENERIC and "Content..." not in c, r["uuid"]
        have.add(r["uuid"])
        rows.append(r)
        added += 1
print("staging incorporado:", added)

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(
    os.path.join(ROOT, "checkpoints", "qwen3loop_tool_sft_final"), local_files_only=True)
for r in rows:
    r["token_length"] = len(tok.apply_chat_template(
        r["messages"], tools=r["tools"] or None, tokenize=True, return_dict=False))

with open(PUR, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

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
