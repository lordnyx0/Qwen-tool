import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('checkpoints/qwen3loop_sft_final')
with open('data/sft_tool_calling_curated.jsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10: break
        s = json.loads(line)
        msgs = s['messages']
        tools = s.get('tools') or None
        
        prompt_txt = tok.apply_chat_template(msgs[:-1], tools=tools, add_generation_prompt=True, tokenize=False)
        full_txt = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=False, tokenize=False)
        
        assert full_txt.startswith(prompt_txt), f"Sample {s['id']} failed alignment!"
        prompt_ids = tok.encode(prompt_txt, add_special_tokens=False)
        full_ids = tok.encode(full_txt, add_special_tokens=False)
        
        # Check target tokens
        target_ids = full_ids[len(prompt_ids):]
        decoded_target = tok.decode(target_ids)
        preview = repr(decoded_target[:60])
        print(f"Sample {s['id']} (Diff {s['difficulty']}): Target len = {len(target_ids)} tokens | Preview: {preview}")

print("\nSanity check passed! 100% aligned.")
