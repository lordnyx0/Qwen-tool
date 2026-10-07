import json

with open('data/eval_90_battery_tool_sft_q8_results.json', encoding='utf-8') as f:
    d = json.load(f)

print(f"Total: {d['total_cases']} | Pass: {d['total_pass']} ({d['overall_accuracy_pct']}%)")
print("Breakdown por categoria:")
for k, v in d['category_breakdown'].items():
    print(f"  {k}: {v['pass']}/{v['total']} (pass) | format: {v['format']}/{v['total']}")

print("\nPrimeiros 10 resultados:")
for r in d['detailed_results'][:10]:
    p = "PASS" if r['eval']['overall_pass'] else "FAIL"
    print(f"[{p}] ID: {r['id']} | Cat: {r['category']}")
    print(f"  Prompt: {r['prompt_user'][:80]}")
    print(f"  Output: {repr(r['model_output'])}")
    print(f"  Eval: {r['eval']}")
    print("-" * 60)
