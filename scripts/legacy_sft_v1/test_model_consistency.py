"""
test_model_consistency.py
Testa o modelo qwen3loop-0.6b carregado do model_src (safetensors) com PyTorch/transformers.
Verifica:
  1. Carregamento sem erros
  2. Forward pass (logits shape correto)
  3. Geracao de texto coerente em PT e EN
  4. Ausencia de loops/repeticoes
  5. Aritmetica basica
  6. Contagem de tokens/performance
"""

import sys, time, re, math
sys.path.insert(0, r"C:\Users\Nyx\Downloads\Diversos\qwen\model_src")

import torch
from transformers import AutoTokenizer

# Registrar arch customizada
import qwen3loop_python  # noqa: F401 -- auto-registra no AutoConfig/AutoModel
from transformers import AutoModelForCausalLM, AutoConfig

MODEL_DIR = r"C:\Users\Nyx\Downloads\Diversos\qwen\model_src"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 65)
print(" QWEN3LOOP 0.6B - TESTE DE CONSISTENCIA")
print("=" * 65)
print(f"Device : {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# ── 1. Carregamento ────────────────────────────────────────────
print("\n[1/5] Carregando tokenizer e modelo...")
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(MODEL_DIR, trust_remote_code=True)
print(f"  Config: {cfg.model_type}, layers={cfg.num_hidden_layers}, "
      f"loops={cfg.num_loops}, middle={cfg.loop_middle_layers}, "
      f"split={cfg.enable_double_loop_split}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True,
)
model.eval()
t_load = time.time() - t0
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"  OK! {params:.1f}M parametros em {t_load:.1f}s")

# ── 2. Forward pass ────────────────────────────────────────────
print("\n[2/5] Forward pass (shape check)...")
dummy = tok("Ola mundo", return_tensors="pt").to(DEVICE)
with torch.no_grad():
    out = model(**dummy)
expected_vocab = cfg.vocab_size
assert out.logits.shape[-1] == expected_vocab, \
    f"Vocab size errado: {out.logits.shape[-1]} != {expected_vocab}"
print(f"  OK! logits shape = {tuple(out.logits.shape)}")

# ── 3. Testes de geracao ───────────────────────────────────────
def generate(prompt_text, max_new=150, temp=0.0):
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(
            ids,
            max_new_tokens=max_new,
            do_sample=(temp > 0),
            temperature=temp if temp > 0 else None,
            repetition_penalty=1.05,
        )
    dt = time.time() - t0
    new_ids = gen[0][ids.shape[1]:]
    text_out = tok.decode(new_ids, skip_special_tokens=True)
    tps = len(new_ids) / dt
    return text_out.strip(), len(new_ids), tps

print("\n[3/5] Testes de geracao:")

TESTS = [
    ("Math: 17 * 24",  "What is 17 * 24?",              lambda r: "408" in r),
    ("Math: sqrt(2)",  "What is the square root of 2? Give decimal.", lambda r: "1.41" in r),
    ("Stop: YES/NO",   "Is Paris the capital of France? Answer only YES or NO.", lambda r: ("yes" in r.lower() or "no" in r.lower()) and len(r.split()) <= 10),
    ("PT: saudacao",   "Qual e a capital do Brasil?",    lambda r: "Brasilia" in r or "Bras" in r),
    ("Loop: repeticao","List 3 prime numbers.",          None),  # only check anti-loop
]

all_ok = True
results = []
for name, q, check in TESTS:
    print(f"\n  >> {name}")
    print(f"     Pergunta: {q}")
    answer, n_tok, tps = generate(q, max_new=120)
    print(f"     Resposta: {answer[:200]}")
    print(f"     Tokens  : {n_tok}  |  {tps:.1f} tok/s")

    # Anti-loop check: nenhuma sequencia de 5+ palavras repetida
    words = answer.split()
    loop_detected = False
    if len(words) > 10:
        for i in range(len(words)-5):
            phrase = " ".join(words[i:i+4])
            rest   = " ".join(words[i+4:])
            if phrase in rest:
                loop_detected = True
                break
    if loop_detected:
        print("     [FAIL] Loop detectado na resposta!")
        all_ok = False
    elif check and not check(answer):
        print(f"     [FAIL] Resposta incorreta/inesperada")
        all_ok = False
    else:
        print("     [OK]")

    results.append((name, answer, n_tok, tps, not loop_detected))

# ── 4. Resumo ──────────────────────────────────────────────────
print("\n" + "=" * 65)
if all_ok:
    print("  RESULTADO: TODOS OS TESTES PASSARAM!")
else:
    print("  RESULTADO: ALGUNS TESTES FALHARAM (ver acima)")
print("=" * 65)

avg_tps = sum(r[3] for r in results) / len(results)
print(f"  Media de velocidade: {avg_tps:.1f} tok/s no {DEVICE.upper()}")
print()
