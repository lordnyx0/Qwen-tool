import sys
sys.stdout.reconfigure(encoding='utf-8')
from datasets import load_dataset_builder

repos = [
    'bellfire/openclaw-coder-dataset',
    'sroecker/hermes-agent-traces-chatml',
    'NousResearch/hermes-function-calling-v1',
    'nvidia/Nemotron-Terminal-Corpus'
]

print("=== TAMANHOS E LINHAS DOS DATASETS ===")
for r in repos:
    try:
        builder = load_dataset_builder(r)
        splits = builder.info.splits
        print(f"\nRepo: {r}")
        print(f"Splits: {splits}")
        print(f"Features: {list(builder.info.features.keys()) if builder.info.features else None}")
    except Exception as e:
        print(f"\nRepo: {r} -> Info não disponível direto: {e}")
