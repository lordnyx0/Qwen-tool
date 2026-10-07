import json
import os
from collections import Counter

def scan_local_file(path):
    print(f"\n{'='*70}\nSCANNING: {path}\n{'='*70}")
    if not os.path.exists(path):
        print("File does not exist.")
        return
    total = 0
    t1_has_tool = 0
    t1_text_only = 0
    tool_counter = Counter()
    roles_seq = Counter()
    first_asst_snippets = []
    
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            item = json.loads(line)
            msgs = item.get("messages", [])
            tools = item.get("tools", [])
            for t in tools:
                fn = t.get("function", t)
                tool_counter[fn.get("name", "unk")] += 1
                
            r_seq = " -> ".join([m.get("role", "") for m in msgs])
            roles_seq[r_seq] += 1
            
            assts = [m for m in msgs if m.get("role") == "assistant"]
            if assts:
                first = assts[0]
                if first.get("tool_calls"):
                    t1_has_tool += 1
                else:
                    t1_text_only += 1
                    if len(first_asst_snippets) < 5:
                        first_asst_snippets.append(first.get("content", "")[:120])
                        
    print(f"Total samples: {total}")
    print(f"Turn 1 tool call: {t1_has_tool} ({t1_has_tool/total*100:.1f}%)")
    print(f"Turn 1 text only: {t1_text_only} ({t1_text_only/total*100:.1f}%)")
    print(f"Unique tools: {len(tool_counter)}")
    print(f"Top 10 tools: {tool_counter.most_common(10)}")
    print(f"Top 3 role sequences:")
    for seq, count in roles_seq.most_common(3):
        print(f"   [{count}]: {seq[:100]}...")
    if first_asst_snippets:
        print("Sample Turn 1 text:")
        for s in first_asst_snippets[:3]:
            print(f"   \"{s}\"")

scan_local_file("data/sft_tool_calling_curated.jsonl")
scan_local_file("data/sft_tool_calling_v2.jsonl")
