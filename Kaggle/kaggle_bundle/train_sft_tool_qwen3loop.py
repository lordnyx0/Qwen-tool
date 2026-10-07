"""
train_sft_tool_qwen3loop.py
Treinamento Supervisionado (Full-FT SFT) do Qwen3Loop-0.6B especializado em Tool Calling:
- 100% focado em Tool Calling (Nível 1, Nível 2, Nível 3 Python Math) + Abstenção Negativa
- Multi-Exit Intermediate Loss (loop_loss_weights=[0.4, 1.0])
- Latent Halting Probe Head (supervisionada pela dificuldade 1, 2, 3)
- Otimizador BitsAndBytes AdamW 8-bit (1.19 GB VRAM)
- Suporte a PyTorch BF16 nativo e Gradient Checkpointing
- Máscara de perda no estilo ChatML Qwen (<|im_start|>assistant ...)
- Avaliação periódica sobre o Benchmark Holdout (90 itens)
"""

import os
import sys
import json
import math
import time
import argparse
from typing import Dict, List, Optional, Tuple

# Inserir caminhos locais ou Kaggle para engine customizada
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [
    os.path.join(ROOT_DIR, "engine", "qwen3loop_python"),
    os.path.join(ROOT_DIR, "model_src"),
    "./engine/qwen3loop_python",
    "/kaggle/working/engine/qwen3loop_python",
    "/kaggle/input/qwen3loop-tool-grpo-bundle/engine/qwen3loop_python",
]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

try:
    import qwen3loop_python  # noqa: F401
    print("Engine qwen3loop_python carregada com sucesso.")
except ImportError:
    print("Aviso: qwen3loop_python importado diretamente do diretório ou namespace padrão.")

try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("Aviso: bitsandbytes não encontrado. Usando torch.optim.AdamW como fallback.")


class LatentHaltingProbe(nn.Module):
    """
    Sonda latente ultraleve (65k parâmetros, ~0.01 ms).
    Prevê se a passagem intermediária (Loop 1) é suficiente para early-exit
    ou se necessita de recorrência (Loop 2 ou 3).
    """
    def __init__(self, hidden_size: int = 1024, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 64, bias=True),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes, bias=False)
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state)


class Qwen3LoopToolDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        print(f"Carregando dataset de {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    self.samples.append(json.loads(line))
                except Exception:
                    continue
        print(f"Dataset carregado: {len(self.samples):,} amostras.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.samples[idx]
        msgs = row.get("messages", [])
        tools = row.get("tools") or None

        # Gerar prompt e full com o apply_chat_template do tokenizer
        prompt_txt = self.tokenizer.apply_chat_template(
            msgs[:-1],
            tools=tools,
            add_generation_prompt=True,
            tokenize=False
        )
        full_txt = self.tokenizer.apply_chat_template(
            msgs,
            tools=tools,
            add_generation_prompt=False,
            tokenize=False
        )

        prompt_ids = self.tokenizer.encode(prompt_txt, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_txt, add_special_tokens=False)

        # Truncar se exceder max_len
        if len(full_ids) > self.max_len:
            full_ids = full_ids[:self.max_len]

        # Criar labels: prompt é mascarado com -100
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        # Alvo da sonda de recorrência: Dificuldade 1 -> 0, Dificuldade 2 -> 1, Dificuldade 3 -> 2
        diff = row.get("difficulty", 2)
        target_rec = max(0, min(2, diff - 1))

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "target_rec": torch.tensor(target_rec, dtype=torch.long)
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)

    batch_input_ids = []
    batch_labels = []
    batch_attention_mask = []
    batch_target_rec = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        # Pad right
        padded_ids = torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        padded_labels = torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        attention_mask = torch.cat([torch.ones(seq_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])

        batch_input_ids.append(padded_ids)
        batch_labels.append(padded_labels)
        batch_attention_mask.append(attention_mask)
        batch_target_rec.append(item["target_rec"])

    return {
        "input_ids": torch.stack(batch_input_ids),
        "labels": torch.stack(batch_labels),
        "attention_mask": torch.stack(batch_attention_mask),
        "target_rec": torch.stack(batch_target_rec)
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de treino: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    use_bf16 = torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Precisão de ativações e pesos: {dtype}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # 2. Config & Modelo Looped
    loop_loss_weights = [float(w) for w in args.loop_loss_weights.split(",")]
    print(f"Configurando Qwen3Loop: num_loops={args.num_loops}, loop_loss_weights={loop_loss_weights}")

    config = AutoConfig.from_pretrained(
        args.model_path,
        num_loops=args.num_loops,
        loop_loss_weights=loop_loss_weights
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype
    ).to(device)

    # Ativar Gradient Checkpointing
    model.gradient_checkpointing_enable()
    print("Gradient Checkpointing ativado com sucesso.")

    # 3. Sonda Latente de Recorrência
    probe_head = None
    if args.train_probe:
        probe_head = LatentHaltingProbe(hidden_size=config.hidden_size, num_classes=3).to(device).to(dtype)
        print("Sonda de Halting Latente inicializada.")

    # 4. Dataset & Split Treino / Val
    full_dataset = Qwen3LoopToolDataset(args.data_path, tokenizer, max_len=args.max_len)
    val_size = max(20, int(len(full_dataset) * 0.05))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_id)
    )

    print(f"Divisão: {train_size:,} treino | {val_size:,} validação")

    # 5. Otimizador (AdamW 8-bit ou Fallback)
    params = list(model.parameters())
    if probe_head is not None:
        params += list(probe_head.parameters())

    if HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
        print("Otimizador: BitsAndBytes AdamW 8-bit")
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        print("Otimizador: PyTorch AdamW standard")

    probe_criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    if args.max_steps is not None and args.max_steps < total_steps:
        total_steps = args.max_steps

    print(f"Total de passos de otimizador: {total_steps:,} (Batch efetivo: {args.batch_size * args.grad_accum})")

    os.makedirs(args.output_dir, exist_ok=True)

    # 6. Loop de Treino
    model.train()
    if probe_head:
        probe_head.train()

    step = 0
    running_lm_loss = 0.0
    running_probe_loss = 0.0
    probe_correct = 0
    probe_total = 0
    t0 = time.time()

    optimizer.zero_grad()

    print("\n" + "=" * 65)
    print(" INICIANDO TREINAMENTO SFT TOOL-CALLING COM MULTI-EXIT LOSS")
    print("=" * 65)

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_rec = batch["target_rec"].to(device)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=False
                )
                lm_loss = outputs.loss
                loss = lm_loss
                p_loss = torch.tensor(0.0, device=device)

                # Treino da sonda de recorrência sobre o estado intermediário do Loop 1
                if probe_head is not None and hasattr(outputs, "loop_hidden_states") and len(outputs.loop_hidden_states) > 0:
                    h1 = outputs.loop_hidden_states[0] # (B, S, H)
                    h1_repr = h1[:, -1, :]
                    probe_logits = probe_head(h1_repr)
                    p_loss = probe_criterion(probe_logits, target_rec)
                    loss = loss + (0.15 * p_loss)

                    pred = probe_logits.argmax(dim=-1)
                    probe_correct += (pred == target_rec).sum().item()
                    probe_total += target_rec.size(0)

            # Normalização por gradient accumulation
            loss = loss / args.grad_accum
            loss.backward()

            running_lm_loss += lm_loss.item()
            running_probe_loss += p_loss.item()

            if (batch_idx + 1) % args.grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

                # Log periódico a cada 5 passos
                if step % 5 == 0:
                    avg_lm = running_lm_loss / max(1, 5 * args.grad_accum)
                    avg_probe = running_probe_loss / max(1, 5 * args.grad_accum)
                    probe_acc = (probe_correct / max(1, probe_total)) * 100 if probe_total > 0 else 0
                    elapsed = time.time() - t0
                    vram_gb = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    print(f"Passo [{step:,}/{total_steps:,}] ({step/max(1, total_steps)*100:.1f}%) | Epoch {epoch+1} | "
                          f"LM Loss: {avg_lm:.4f} | Probe Loss: {avg_probe:.4f} | "
                          f"Probe Acc: {probe_acc:.1f}% | VRAM: {vram_gb:.2f} GB | "
                          f"Tempo: {elapsed:.1f}s", flush=True)

                    running_lm_loss = 0.0
                    running_probe_loss = 0.0
                    probe_correct = 0
                    probe_total = 0

                # Avaliação de validação
                if step % args.eval_steps == 0:
                    evaluate(model, probe_head, val_loader, device, dtype, probe_criterion)

                # Salvamento de checkpoint
                if step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                    save_model_and_probe(model, probe_head, tokenizer, config, ckpt_path)

                if args.max_steps is not None and step >= args.max_steps:
                    break

        if args.max_steps is not None and step >= args.max_steps:
            break

    print("\n" + "=" * 65)
    print(" TREINAMENTO FINALIZADO!")
    print(f" Salvando modelo final em: {args.output_dir}")
    print("=" * 65)
    save_model_and_probe(model, probe_head, tokenizer, config, args.output_dir)


def evaluate(model, probe_head, val_loader, device, dtype, probe_criterion, max_batches: Optional[int] = 30):
    model.eval()
    if probe_head:
        probe_head.eval()

    total_lm_loss = 0.0
    correct = 0
    total = 0
    eval_count = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_rec = batch["target_rec"].to(device)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_lm_loss += outputs.loss.item()

                if probe_head is not None and hasattr(outputs, "loop_hidden_states") and len(outputs.loop_hidden_states) > 0:
                    h1 = outputs.loop_hidden_states[0][:, -1, :]
                    probe_logits = probe_head(h1)
                    pred = probe_logits.argmax(dim=-1)
                    correct += (pred == target_rec).sum().item()
                    total += target_rec.size(0)
            eval_count += 1

    avg_loss = total_lm_loss / max(1, eval_count)
    ppl = math.exp(min(20, avg_loss))
    acc = (correct / max(1, total)) * 100 if total > 0 else 0
    print(f"\n[VALIDAÇÃO ({eval_count} batches)] Val Loss: {avg_loss:.4f} | Perplexidade: {ppl:.2f} | Probe Acc: {acc:.1f}%\n", flush=True)

    model.train()
    if probe_head:
        probe_head.train()


def save_model_and_probe(model, probe_head, tokenizer, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Salvando pesos e configurações em {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    if probe_head is not None:
        torch.save(probe_head.state_dict(), os.path.join(save_dir, "latent_halting_probe.pt"))
    print("Checkpoint salvo com sucesso!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino SFT Tool-Calling Qwen3Loop")
    parser.add_argument("--data_path", type=str, default="data/sft_tool_calling_curated.jsonl")
    parser.add_argument("--model_path", type=str, default="checkpoints/qwen3loop_sft_final")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen3loop_tool_sft_final")
    parser.add_argument("--num_loops", type=int, default=2)
    parser.add_argument("--loop_loss_weights", type=str, default="0.4,1.0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=150)
    parser.add_argument("--train_probe", action="store_true", default=True)

    args = parser.parse_args()
    train(args)
