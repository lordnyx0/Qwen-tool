import sys
sys.stdout.reconfigure(encoding='utf-8')
from datasets import load_dataset

print("=== INSPECIONANDO nvidia/Nemotron-Terminal-Corpus ===")
for cfg in ['skill_based_easy', 'skill_based_medium', 'dataset_adapters']:
    print(f"\n--- CONFIG: {cfg} ---")
    try:
        ds = load_dataset("nvidia/Nemotron-Terminal-Corpus", cfg, split="train", streaming=True)
        for i, row in enumerate(ds):
            if i >= 2: break
            print(f"[Amostra {i+1}]")
            for k in list(row.keys()):
                val = str(row[k])
                print(f"  {k}: {val[:120]}...")
    except Exception as e:
        print(f"Erro em {cfg}: {e}")
