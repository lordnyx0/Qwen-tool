import sys
sys.stdout.reconfigure(encoding='utf-8')
from datasets import load_dataset
from huggingface_hub import HfApi

api = HfApi()

# 1. Checar Hermes
print("--- BUSCANDO REPOS HERMES NO HF ---")
hermes_repos = api.list_datasets(search="hermes-function-calling")
for r in hermes_repos:
    print("Found:", r.id)

# 2. Inspecionar OpenClaw em UltraData (General-Agent)
print("\n--- INSPECIONANDO OPENCLAW (General-Agent) ---")
try:
    ds_gen = load_dataset("openbmb/UltraData-SFT-Agent-2609", "General-Agent", split="train", streaming=True)
    for i, row in enumerate(ds_gen):
        if i >= 2: break
        uuid = row.get("uuid")
        source = row.get("source")
        domain = row.get("domain")
        tools = row.get("tools", [])
        msgs = row.get("messages", [])
        print(f"\n[Sample {i+1}] UUID: {uuid} | Source: {source} | Domain: {domain}")
        print(f"Tools ({len(tools)}): {[t.get('function', {}).get('name') for t in tools[:6]]}")
        for m_idx, m in enumerate(msgs[:4]):
            role = m.get("role")
            content = str(m.get("content") or "")[:90].replace("\n", " ")
            tc = [c.get("function", {}).get("name") for c in m.get("tool_calls", [])] if m.get("tool_calls") else None
            print(f"  [{m_idx}] {role.upper():10} | calls: {tc} | content: {content}")
except Exception as e:
    print(f"Erro em General-Agent: {e}")
