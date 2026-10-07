import sys
import json
sys.stdout.reconfigure(encoding='utf-8')
from datasets import load_dataset

# Inspecionar OpenClaw Coder
print("=" * 70)
print("AMOSTRA COMPLETA: bellfire/openclaw-coder-dataset")
print("=" * 70)
try:
    ds_claw = load_dataset('bellfire/openclaw-coder-dataset', split='train', streaming=True)
    sample = next(iter(ds_claw))
    msgs = sample.get('messages', [])
    for idx, m in enumerate(msgs[:6]):
        role = m.get('role')
        content = str(m.get('content') or '')[:120].replace('\n', ' ')
        print(f"[{idx}] {role.upper():10} | content: {content}")
except Exception as e:
    print("Erro:", e)

# Inspecionar Hermes Agent Traces ChatML
print("\n" + "=" * 70)
print("AMOSTRA COMPLETA: sroecker/hermes-agent-traces-chatml")
print("=" * 70)
try:
    ds_h = load_dataset('sroecker/hermes-agent-traces-chatml', split='train', streaming=True)
    sample = next(iter(ds_h))
    msgs = sample.get('messages', [])
    for idx, m in enumerate(msgs[:6]):
        role = m.get('role')
        content = str(m.get('content') or '')[:120].replace('\n', ' ')
        print(f"[{idx}] {role.upper():10} | content: {content}")
except Exception as e:
    print("Erro:", e)
