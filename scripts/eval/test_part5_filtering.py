import os
import sys
import json
import time
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(".")
TOKENIZER_DIR = os.path.join(ROOT_DIR, "checkpoints", "qwen3loop_sft_final")

print("Carregando tokenizer Qwen...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)

cache_dir = r"C:\Users\Nyx\.cache\huggingface\hub\datasets--openbmb--UltraData-SFT-Agent-2609\snapshots\f684cc1a9f3e19f6f4929102cd9b06cc0b895b8a\data\Tool_Use"
part5_path = os.path.join(cache_dir, "Tool_Use_part-5-of-9.jsonl")

print(f"Testando filtro em {part5_path}...")
count = 0
passed = 0
too_long = 0
render_err = 0

t0 = time.time()
with open(part5_path, "r", encoding="utf-8") as f:
    for line in f:
        count += 1
        item = json.loads(line)
        msgs = item.get("messages", [])
        tools = item.get("tools", [])
        
        # Pre-filter by char length
        total_chars = sum(len(m.get("content") or "") for m in msgs)
        if total_chars > 25000:
            too_long += 1
            continue
            
        # Normalizar tool_calls em msgs
        for m in msgs:
            if m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    if isinstance(fn.get("arguments"), dict):
                        fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
                        
        try:
            rendered = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False)
            tokens = tokenizer.encode(rendered, add_special_tokens=False)
            t_len = len(tokens)
            if t_len <= 5000:
                passed += 1
            else:
                too_long += 1
        except Exception as e:
            render_err += 1

dt = time.time() - t0
print(f"Part 5 processada em {dt:.1f}s:")
print(f"  Total: {count}")
print(f"  <= 5000 tokens: {passed} ({passed/count*100:.1f}%)")
print(f"  > 5000 tokens: {too_long} ({too_long/count*100:.1f}%)")
print(f"  Erros: {render_err}")
