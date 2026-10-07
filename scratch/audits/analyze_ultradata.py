import json
import sys
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

srcs = Counter()
has_tc_count = Counter()
dict_type_count = 0
total = 0

with open('data/cleaned/ultradata_toolmind_curated_16k.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        total += 1
        item = json.loads(line)
        s = item.get('source', 'unknown')
        srcs[s] += 1
        
        msgs = item.get('messages', [])
        has_tc = any(m.get('tool_calls') for m in msgs if isinstance(m, dict))
        if has_tc:
            has_tc_count[s] += 1
            
        t_str = json.dumps(item.get('tools', []))
        if '"type": "dict"' in t_str or '"type":"dict"' in t_str:
            dict_type_count += 1

print(f"Total: {total:,}")
for s, c in srcs.items():
    print(f"  {s}: {c:,} total | {has_tc_count[s]:,} com tool_calls ({has_tc_count[s]/c*100:.1f}%)")
print(f"Amostras com 'type': 'dict': {dict_type_count:,}")
