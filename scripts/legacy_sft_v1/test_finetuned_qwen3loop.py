"""
test_finetuned_qwen3loop.py
Testa a geração de texto e o comportamento agêntico/código do Qwen3Loop pós-SFT.
"""

import sys
sys.path.insert(0, r"C:\Users\Nyx\Downloads\Diversos\qwen\model_src")

import time
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import qwen3loop_python  # noqa: F401

MODEL_PATH = r"C:\Users\Nyx\Downloads\Diversos\qwen\checkpoints\qwen3loop_sft_final"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Dispositivo: {DEVICE}")
print(f"Carregando modelo treinado de {MODEL_PATH}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
config = AutoConfig.from_pretrained(MODEL_PATH)

dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=dtype).to(DEVICE)
model.eval()

PROMPTS = [
    {
        "desc": "Teste 1: Raciocínio Agêntico / Problema com restrições",
        "system": "When tackling complex reasoning tasks, you have access to the following actions. Use them as needed to progress through your thought process: [ASSESS], [ADVANCE], [VERIFY], [OUTPUT].",
        "user": "Há 5 envelopes na mesa numerados de 1 a 5, e 5 cartas numeradas de 1 a 5. Se distribuirmos aleatoriamente uma carta em cada envelope, qual é a probabilidade de exatamente 4 cartas ficarem no envelope correto? Explique o raciocínio."
    },
    {
        "desc": "Teste 2: Algoritmo de Programação (Python)",
        "system": "You are an expert autonomous software engineer. Think carefully before writing code.",
        "user": "Escreva uma função Python eficiente `def find_longest_valid_parentheses(s: str) -> int:` que encontra o comprimento da maior substring contendo parênteses bem formados. Complexidade de tempo O(N) e espaço O(1) ou O(N)."
    }
]

for p in PROMPTS:
    print("\n" + "=" * 70)
    print(f"{p['desc']}")
    print("=" * 70)
    print(f"Pergunta do Usuário:\n{p['user']}\n")

    chat = [
        {"role": "system", "content": p["system"]},
        {"role": "user", "content": p["user"]}
    ]
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id
        )
    t_gen = time.time() - t0

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    n_tokens = len(generated_ids)
    tok_per_sec = n_tokens / max(1e-4, t_gen)

    print(f"Resposta Gerada ({n_tokens} tokens, {tok_per_sec:.1f} tokens/s):\n")
    print(response)

print("\n" + "=" * 70)
print("TESTE DE GERAÇÃO CONCLUÍDO COM SUCESSO!")
print("=" * 70)
