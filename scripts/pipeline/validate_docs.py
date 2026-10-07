"""
validate_docs.py — auditoria mecânica doc<->código (padrão industrial).
Uso: python scripts/pipeline/validate_docs.py [pasta_docs]
Checa: links file:/// + paths em code spans (scripts/, scratch/, data/, kaggle_*)
       apontam para arquivos existentes. Exit 0 se zero quebras.
Menções intencionais a arquivos inexistentes devem ser escritas SEM path
(ex.: "o `poll_logs` citado em versões antigas não existe") para não falhar aqui.
"""
import os
import re
import sys
from glob import glob

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DOCS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "docs")

fails = []
checked = 0
targets = sorted(glob(os.path.join(DOCS, "**", "*.md"), recursive=True))
for extra in ("README.md", "AGENTS.md", os.path.join("data", "README.md"),
              os.path.join("scripts", "README.md"), os.path.join("scratch", "README.md")):
    p = os.path.join(ROOT, extra)
    if p not in targets and os.path.exists(p):
        targets.append(p)
for fp in targets:
    text = open(fp, encoding="utf-8").read()
    for m in re.finditer(r"file:///c:/Users/Nyx/Downloads/Diversos/qwen/(\S+?)([\"')\s>])", text):
        rel = m.group(1)
        checked += 1
        if not os.path.exists(os.path.join(ROOT, rel)):
            fails.append((fp, "file:// morto: " + rel))
    for pat in (r"`(scripts/[\w/.-]+\.py)`", r"`(scratch/[\w/.-]+\.py)`",
                r"`(data/[\w/.-]+\.jsonl)`", r"`(kaggle_\w+/[\w/.-]+)`"):
        for m in re.finditer(pat, text):
            rel = m.group(1)
            checked += 1
            if not os.path.exists(os.path.join(ROOT, rel)):
                fails.append((fp, "path citado inexistente: " + rel))

print("referencias checadas:", checked)
print("quebras:", len(fails))
for fp, msg in fails:
    print(" -", os.path.relpath(fp, ROOT), "|", msg)
sys.exit(0 if not fails else 1)
