import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import qwen3loop_python
from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM
from scripts.train_sft_tool_qwen3loop import Qwen3LoopToolDataset, collate_fn, LatentHaltingProbe

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = "checkpoints/qwen3loop_sft_final"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_path, num_loops=2, loop_loss_weights=[])
    dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="sdpa"
    ).to(device)

    model.gradient_checkpointing_enable()
    probe = LatentHaltingProbe(hidden_size=config.hidden_size, num_classes=3).to(device).to(dtype)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(probe.parameters()), lr=1e-5)

    dataset = Qwen3LoopToolDataset("data/sft_ultradata_tool_use_8k.jsonl", tokenizer, max_len=4096)
    
    # Encontrar a amostra mais longa do dataset
    max_idx = 0
    max_len = 0
    for idx in range(min(500, len(dataset))):
        l = dataset[idx]["input_ids"].size(0)
        if l > max_len:
            max_len = l
            max_idx = idx

    print(f"Amostra mais longa encontrada: idx={max_idx}, len={max_len}")
    sample = dataset[max_idx]
    batch = collate_fn([sample], tokenizer.pad_token_id)

    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    target_rec = batch["target_rec"].to(device)

    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad()

    with torch.autocast(device_type=device.type, dtype=dtype):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        if hasattr(outputs, "loop_hidden_states") and outputs.loop_hidden_states:
            h1 = outputs.loop_hidden_states[0]
            last_idx = attention_mask.sum(dim=-1) - 1
            h1_repr = h1[torch.arange(h1.size(0), device=h1.device), last_idx]
            probe_logits = probe(h1_repr)
            p_loss = nn.CrossEntropyLoss()(probe_logits, target_rec)
            loss = loss + 0.15 * p_loss

    print(f"Forward OK! Loss={loss.item():.4f}")
    loss.backward()
    print("Backward OK!")
    optimizer.step()
    print("Optimizer Step OK!")

    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"PICO DE VRAM COM BATCH=1 E LEN={max_len}: {peak_gb:.2f} GB")

if __name__ == "__main__":
    main()
