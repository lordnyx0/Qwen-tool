import json
import os
import re
from collections import Counter, defaultdict

dataset_path = "data/sft_ultradata_tool_use_clean_5k.jsonl"

total_samples = 0
turn_counts = Counter()
turn1_actions = Counter()
turn1_text_patterns = Counter()
turn1_assistant_samples = []

first_tool_turn = Counter()
tool_names_counter = Counter()
domains_counter = Counter()

# Patterns to detect in turn 1 text
patterns = {
    "ask_email": re.compile(r"\b(email|e-mail)\b", re.I),
    "ask_phone": re.compile(r"\b(phone|telephone|mobile)\b", re.I),
    "ask_name": re.compile(r"\b(full name|your name|first and last name)\b", re.I),
    "verify_identity": re.compile(r"\b(verify your identity|verification|verify you|authentication|authenticate|confirm your identity)\b", re.I),
    "ask_account_id": re.compile(r"\b(account id|account number|customer id|user id|membership id)\b", re.I),
    "ask_order_id": re.compile(r"\b(order id|order number|tracking number|tracking id)\b", re.I),
    "ask_clarification": re.compile(r"\b(could you please provide|can you provide|please specify|which one|more details)\b", re.I),
    "apology_or_greeting": re.compile(r"\b(i'd be happy to|hello|hi|welcome to|thank you for reaching out)\b", re.I),
}

with open(dataset_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        total_samples += 1
        item = json.loads(line)
        messages = item.get("messages", [])
        tools = item.get("tools", [])
        
        for t in tools:
            fn = t.get("function", t)
            name = fn.get("name", "unknown")
            tool_names_counter[name] += 1
            
        # Count user/assistant turns
        dialog_turns = [m for m in messages if m.get("role") in ["user", "assistant"]]
        turn_counts[len(dialog_turns)] += 1
        
        # Analyze first assistant turn
        first_asst = None
        first_asst_idx = -1
        first_tool_call_idx = None
        
        asst_count = 0
        for m_idx, m in enumerate(messages):
            if m.get("role") == "assistant":
                asst_count += 1
                if first_asst is None:
                    first_asst = m
                    first_asst_idx = asst_count
                if m.get("tool_calls") and first_tool_call_idx is None:
                    first_tool_call_idx = asst_count

        if first_tool_call_idx is not None:
            first_tool_turn[first_tool_call_idx] += 1
        else:
            first_tool_turn["never"] += 1
            
        if first_asst:
            if first_asst.get("tool_calls"):
                turn1_actions["direct_tool_call"] += 1
            else:
                turn1_actions["pure_text"] += 1
                content = first_asst.get("content", "")
                
                matched = False
                for p_name, p_regex in patterns.items():
                    if p_regex.search(content):
                        turn1_text_patterns[p_name] += 1
                        matched = True
                        
                if not matched:
                    turn1_text_patterns["other_text"] += 1
                    
                if len(turn1_assistant_samples) < 15:
                    # Collect first user message and first assistant message
                    first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
                    turn1_assistant_samples.append({
                        "user": first_user[:120],
                        "asst": content[:200]
                    })

print("="*60)
print(f"ANALYSIS OF DATASET: {dataset_path}")
print(f"Total samples: {total_samples}")
print("="*60)

print("\n--- FIRST ASSISTANT ACTION ---")
for action, count in turn1_actions.items():
    pct = (count / total_samples) * 100
    print(f"  {action}: {count} ({pct:.1f}%)")

print("\n--- WHEN DOES THE FIRST TOOL CALL OCCUR (Assistant Turn #) ---")
for turn_num, count in sorted(first_tool_turn.items(), key=lambda x: str(x[0])):
    pct = (count / total_samples) * 100
    print(f"  Turn {turn_num}: {count} ({pct:.1f}%)")

print("\n--- TURN 1 TEXT CLASSIFICATION (When Turn 1 is Text) ---")
for p_name, count in turn1_text_patterns.most_common():
    pct = (count / total_samples) * 100
    print(f"  {p_name}: {count} ({pct:.1f}%)")

print("\n--- TOP 25 TOOLS IN DATASET ---")
for name, count in tool_names_counter.most_common(25):
    print(f"  {name}: {count}")

print("\n--- SAMPLE TURN 1 EXCHANGES (Pure Text) ---")
for idx, s in enumerate(turn1_assistant_samples[:8], 1):
    print(f"\n[Sample {idx}]")
    print(f"USER: {s['user']}")
    print(f"ASST: {s['asst']}")
