import os
import sys
import torch
import torch.nn as nn

# Ensure engine is in path
sys.path.insert(0, os.path.abspath("."))

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import qwen3loop_python
from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM
from scripts.train_sft_tool_qwen3loop import Qwen3LoopToolDataset, collate_fn, LatentHaltingProbe

def main():
    print("=== TESTE EMPÍRICO DE VALIDAÇÃO PRÉ-KAGGLE ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de teste: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    model_path = "checkpoints/qwen3loop_sft_final"
    if not os.path.exists(model_path):
        model_path = "model_src"

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_path, num_loops=2, loop_loss_weights=[])
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Carregando modelo de {model_path} em {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="sdpa"
    ).to(device)

    model.gradient_checkpointing_enable()
    probe = LatentHaltingProbe(hidden_size=config.hidden_size, num_classes=3).to(device).to(dtype)
    probe_criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(list(model.parameters()) + list(probe.parameters()), lr=1e-5)

    dataset = Qwen3LoopToolDataset("data/sft_ultradata_tool_use_8k.jsonl", tokenizer, max_len=4096)
    # Pegar 2 amostras
    batch_raw = [dataset[0], dataset[1]]
    batch = collate_fn(batch_raw, tokenizer.pad_token_id)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_rec = batch["target_rec"].to(device)

    print(f"Batch shape: {input_ids.shape}, tokens assistente com loss: {(labels != -100).sum().item()}")

    optimizer.zero_grad()
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=torch.cuda.is_available()):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        lm_loss = outputs.loss
        loss = lm_loss

        if hasattr(outputs, "loop_hidden_states") and outputs.loop_hidden_states:
            h1 = outputs.loop_hidden_states[0]
            last_idx = attention_mask.sum(dim=-1) - 1
            h1_repr = h1[torch.arange(h1.size(0), device=h1.device), last_idx]
            probe_logits = probe(h1_repr)
            p_loss = probe_criterion(probe_logits, target_rec)
            loss = loss + 0.15 * p_loss

    print(f"Loss forward calculada: LM={lm_loss.item():.4f}, Total={loss.item():.4f}")
    loss.backward()

    # Verificar gradientes de todas as camadas unrolled
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()

    print(f"Total de parâmetros com gradientes não-nulos: {len(grad_norms)} / {len(list(model.parameters()))}")

    # Checar camadas de cauda (layers 21 a 27 do loop 2)
    tail_layers_with_grad = [n for n in grad_norms if "layers.25" in n or "layers.26" in n or "layers.27" in n]
    print(f"Gradientes presentes nas camadas 25-27: {len(tail_layers_with_grad)} parâmetros verificados OK")
    assert len(tail_layers_with_grad) > 0, "ERRO: Camadas 25-27 sem gradientes!"

    optimizer.step()
    if torch.cuda.is_available():
        peak_vram = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"Pico de VRAM utilizado: {peak_vram:.2f} GB")

    print("=== VALIDAÇÃO EMPÍRICA CONCLUÍDA COM 100% DE SUCESSO! ===")

if __name__ == "__main__":
    main()
