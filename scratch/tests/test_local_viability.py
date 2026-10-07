import torch
import torch.nn as nn
import os
import sys

# Testar se FlashAttention-2 / SDPA roda na RTX 3060
print(f"CUDA Disponível: {torch.cuda.is_available()}")
device = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
print(f"Dispositivo: {props.name} (CC {props.major}.{props.minor})")
print(f"VRAM Total: {props.total_memory / 1024**3:.2f} GB")
print(f"VRAM Livre Inicial: {torch.cuda.mem_get_info()[0] / 1024**3:.2f} GB")

# Testar suporte dos backends SDPA na RTX 3060 (Ampere CC 8.6)
flash_ok = torch.backends.cuda.flash_sdp_enabled()
mem_ok = torch.backends.cuda.mem_efficient_sdp_enabled()
math_ok = torch.backends.cuda.math_sdp_enabled()
print(f"Backends SDPA ativos -> Flash: {flash_ok} | Mem-Efficient: {mem_ok} | Math: {math_ok}")

# Carregar uma camada do modelo com config oficial
sys.path.insert(0, "engine")
import qwen3loop_python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("checkpoints/qwen3loop_sft_final")
config._attn_implementation = "sdpa"

print("Inicializando modelo de teste na GPU...")
model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/qwen3loop_sft_final",
    config=config,
    torch_dtype=torch.float16,
    attn_implementation="sdpa"
).to(device)

model.gradient_checkpointing_enable()

# Testar 1 forward + backward pass com seq_len = 4096 e 6144
for seq_len in [2048, 4096, 6144]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
    labels = input_ids.clone()
    labels[:, :seq_len - 500] = -100 # 500 tokens ativos
    
    try:
        out = model(input_ids=input_ids, attention_mask=None, labels=labels)
        loss = out.loss
        loss.backward()
        peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"seq_len = {seq_len:4d} | Loss: {loss.item():.4f} | Peak VRAM: {peak_gb:.2f} GB -> SUCESSO!")
    except torch.OutOfMemoryError as e:
        print(f"seq_len = {seq_len:4d} -> OOM: {e}")
        break

print("\nTeste concluído com sucesso!")
