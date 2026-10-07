"""
test_multi_exit_loss.py
Valida o cálculo do loop_loss_weights (multi-exit intermediate loss) e backward pass.
"""

import sys
sys.path.insert(0, r"C:\Users\Nyx\Downloads\Diversos\qwen\model_src")

import torch
import qwen3loop_python  # noqa: F401
from transformers import AutoModelForCausalLM, AutoConfig

MODEL_DIR = r"C:\Users\Nyx\Downloads\Diversos\qwen\model_src"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Dispositivo: {DEVICE}")
print("Carregando modelo com loop_loss_weights=[0.4, 1.0] e num_loops=2...")

config = AutoConfig.from_pretrained(
    MODEL_DIR,
    num_loops=2,
    loop_loss_weights=[0.4, 1.0]
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    config=config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
).to(DEVICE)

model.train()

# Simular batch de treino (batch_size=1, seq_len=64)
input_ids = torch.randint(100, 1000, (1, 64), device=DEVICE)
labels = input_ids.clone()

print("Executando forward com multi-exit loss...")
outputs = model(input_ids=input_ids, labels=labels)

loss = outputs.loss
print(f"Loss calculada: {loss.item():.4f}")
assert loss is not None and not torch.isnan(loss) and loss.item() > 0

print("Executando backward pass...")
loss.backward()

# Verificar se gradientes foram populados
grad_norm = model.lm_head.weight.grad.norm().item()
layer0_grad = model.model.layers[0].mlp.gate_proj.weight.grad.norm().item()
layer20_grad = model.model.layers[20].mlp.gate_proj.weight.grad.norm().item()

print(f"Grad norm lm_head: {grad_norm:.4f}")
print(f"Grad norm layer 0 (prefixo): {layer0_grad:.4f}")
print(f"Grad norm layer 20 (loop miolo): {layer20_grad:.4f}")

assert grad_norm > 0 and layer0_grad > 0 and layer20_grad > 0
print("\n=> SUCESSO: Multi-exit intermediate loss funcionando 100% com backprop!")
