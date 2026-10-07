import json
import time
from llama_cpp import Llama
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained('checkpoints/qwen3loop_tool_sft_final')
llm = Llama('models/gguf/unrolled_modelo_qwen3loop_tool_sft_q8_0.gguf', n_ctx=2048, n_gpu_layers=-1, verbose=False)

with open('data/rl_curriculum/eval_benchmark_battery.jsonl', 'r', encoding='utf-8') as f:
    cases = [json.loads(line) for line in f if line.strip()]

l3_cases = [c for c in cases if c.get('difficulty') == 3][:5]

print(f"Testando {len(l3_cases)} casos de Nível 3 com max_tokens=512 e repeat_penalty=1.05...")

for c in l3_cases:
    prompt = tok.apply_chat_template(c['messages'], tools=c.get('tools'), add_generation_prompt=True, tokenize=False)
    res = llm(prompt, max_tokens=512, temperature=0.0, repeat_penalty=1.05, stop=["<|im_end|>"])
    out = res['choices'][0]['text']
    print("\n" + "=" * 60)
    print("ID:", c['id'])
    print("PROMPT:", c['messages'][-1]['content'])
    print("OUT:\n", out)
