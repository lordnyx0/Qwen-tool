import os
import sys
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath("."))

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import qwen3loop_python
from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM
from scripts.train_sft_tool_qwen3loop import LatentHaltingProbe

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model_path = "checkpoints/qwen3loop_sft_final"
    config = AutoConfig.from_pretrained(model_path, num_loops=2, loop_loss_weights=[])
    dtype = torch.float16

    print("Carregando modelo...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="sdpa"
    ).to(device)

    model.gradient_checkpointing_enable()
    probe = LatentHaltingProbe(hidden_size=config.hidden_size, num_classes=3).to(device).to(dtype)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(probe.parameters()), lr=1e-5)

    seq_len = 3744  # Exatamente o comprimento máximo real do dataset!
    print(f"\nTestando Batch=1 com seq_len={seq_len} (comprimento máximo do dataset)...")

    input_ids = torch.randint(100, 10000, (1, seq_len), device=device)
    attention_mask = torch.ones(1, seq_len, device=device, dtype=torch.long)
    labels = torch.full((1, seq_len), -100, device=device, dtype=torch.long)
    # 200 tokens de assistente ativos para loss
    labels[0, -200:] = input_ids[0, -200:]
    target_rec = torch.tensor([1], device=device, dtype=torch.long)

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
    print(f"\n==========================================")
    print(f"PICO MÁXIMO DE VRAM COM MAX_LEN={seq_len}: {peak_gb:.2f} GB")
    print(f"Capacidade da Tesla T4 (Kaggle): 14.56 GB")
    print(f"Margem de segurança livre: {14.56 - peak_gb:.2f} GB ({((14.56 - peak_gb) / 14.56)*100:.1f}%)")
    print(f"==========================================")

if __name__ == "__main__":
    main()
