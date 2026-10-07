import os
import sys
import json
import time
from huggingface_hub import hf_hub_download

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("INSPECIONANDO ESTRUTURA DOS 9 SHARDS DE Tool-Use (UltraData)")
print("=" * 70)

for part_idx in range(1, 10):
    filename = f"data/Tool_Use/Tool_Use_part-{part_idx}-of-9.jsonl"
    print(f"\nBaixando/Verificando {filename}...")
    t0 = time.time()
    local_path = hf_hub_download(
        "openbmb/UltraData-SFT-Agent-2609",
        filename,
        repo_type="dataset"
    )
    dt = time.time() - t0
    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"-> Arquivo pronto: {size_mb:.1f} MB (baixado em {dt:.1f}s)")
    
    # Inspecionar primeiras 5 linhas
    sources = set()
    sample_tools = []
    line_count = 0
    with open(local_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line_count += 1
            if idx < 10:
                item = json.loads(line)
                sources.add(item.get("source", "unknown"))
                tools = [t.get("function", {}).get("name") for t in item.get("tools", [])]
                sample_tools.extend(tools)
    
    print(f"   Total de linhas: {line_count:,}")
    print(f"   Sources vistos: {sources}")
    print(f"   Exemplos de ferramentas: {sample_tools[:6]}")
