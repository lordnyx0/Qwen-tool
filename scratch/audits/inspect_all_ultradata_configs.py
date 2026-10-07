from datasets import load_dataset
import json

print("=== INSPECTING openbmb/UltraData-SFT-Agent-2609 SUBSETS ===")

for config_name in ['Tool-Use', 'Code-Agent', 'Search-Agent', 'General-Agent']:
    print(f"\n--- CONFIG: {config_name} ---")
    try:
        ds = load_dataset("openbmb/UltraData-SFT-Agent-2609", config_name, split="train", streaming=True)
        count = 0
        for row in ds:
            count += 1
            if count == 1:
                print(f"Keys: {list(row.keys())}")
                uuid = row.get("uuid", "no-uuid")
                tools = row.get("tools", [])
                msgs = row.get("messages", [])
                print(f"Sample 1: uuid={uuid}, len(tools)={len(tools)}, len(msgs)={len(msgs)}")
                # Show first 3 messages
                for idx, m in enumerate(msgs[:4]):
                    role = m.get("role")
                    content = str(m.get("content", ""))[:120].replace("\n", " ")
                    tc = m.get("tool_calls", None)
                    tc_brief = f"calls: {[c.get('function', {}).get('name') for c in tc]}" if tc else "no tc"
                    print(f"  msg[{idx}] {role}: {content} ({tc_brief})")
            if count >= 5:
                break
    except Exception as e:
        print(f"Error loading {config_name}: {e}")
