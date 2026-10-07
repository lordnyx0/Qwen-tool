import json

with open("data/sft_ultradata_tool_use_non_sac_5k.jsonl", "r", encoding="utf-8") as f:
    item = next(json.loads(line) for line in f if json.loads(line).get("source") == "toolmind_graphsyn")
    print(f"UUID: {item.get('uuid')}")
    print(f"Tokens: {item.get('token_length')}")
    print(f"Tools ({len(item.get('tools', []))}): {[t['function']['name'] for t in item.get('tools', [])[:5]]}")
    print("=" * 60)
    for idx, m in enumerate(item["messages"]):
        role = m["role"]
        content = str(m.get("content") or "")[:90].replace("\n", " ")
        tc = [c["function"]["name"] for c in m.get("tool_calls", [])]
        print(f"[{idx}] {role.upper():10} | calls: {tc} | content: {content}")
