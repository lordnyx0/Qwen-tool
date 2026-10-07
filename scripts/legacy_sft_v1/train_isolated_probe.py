"""
train_isolated_probe.py
Treina exclusivamente a cabeça de sonda latente (LatentHaltingProbe)
sobre o modelo Qwen3Loop treinado congelado.
Supervisionado pelos metadados de recorrência gerados pelo Laya.
"""

import os
import sys
import json
import time
import math
import argparse
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, r"C:\Users\Nyx\Downloads\Diversos\qwen\model_src")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import qwen3loop_python  # noqa: F401

MODEL_PATH = r"C:\Users\Nyx\Downloads\Diversos\qwen\checkpoints\qwen3loop_sft_final"
DATA_PATH = r"C:\Users\Nyx\Downloads\Diversos\qwen\data\sft_annotated_curated.jsonl"
OUTPUT_PROBE_PATH = os.path.join(MODEL_PATH, "latent_halting_probe.pt")

class LatentHaltingProbe(nn.Module):
    """
    Sonda latente de roteamento dinâmico (65k parâmetros, ~0.01 ms).
    Recebe o estado oculto da camada 20 (Exit 1 do LoopSplit) no último token do prompt
    e prevê:
    1. logits: classificação (3 classes: 1 loop, 2 loops, 3 loops)
    2. score: índice contínuo de necessidade de deliberação (0.0 a 1.0)
    """
    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        self.classifier = nn.Linear(64, 3, bias=False)
        self.regressor = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(h)
        logits = self.classifier(feat)
        score = self.regressor(feat).squeeze(-1)
        return logits, score


class PromptProbeDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.items = []

        print(f"Carregando dataset de {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                msgs = row.get("messages", [])
                system_txt = ""
                user_txt = ""
                for m in msgs:
                    role = m.get("role")
                    if role == "system" and not system_txt:
                        system_txt = m.get("content", "")
                    elif role == "user" and not user_txt:
                        user_txt = m.get("content", "")

                if not user_txt:
                    user_txt = row.get("prompt", "")
                if not system_txt:
                    system_txt = "You are a helpful and precise reasoning agent."

                meta = row.get("meta", {})
                rec_val = meta.get("recurrence_value", 3)
                target_cls = max(0, min(2, rec_val - 1))
                rec_idx = meta.get("recurrence_index", 0.60)

                prompt_chat = [
                    {"role": "system", "content": system_txt},
                    {"role": "user", "content": user_txt}
                ]
                self.items.append({
                    "chat": prompt_chat,
                    "target_cls": target_cls,
                    "target_score": float(rec_idx)
                })

        print(f"Total de itens carregados: {len(self.items):,}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        return self.items[idx]


def collate_prompt_batch(batch, tokenizer, max_len=512):
    texts = [
        tokenizer.apply_chat_template(item["chat"], tokenize=False, add_generation_prompt=True)
        for item in batch
    ]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    targets_cls = torch.tensor([item["target_cls"] for item in batch], dtype=torch.long)
    targets_score = torch.tensor([item["target_score"] for item in batch], dtype=torch.float)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "target_cls": targets_cls,
        "target_score": targets_score
    }


def train_probe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Carregando Qwen3Loop treinado de {MODEL_PATH}...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, config=config, torch_dtype=dtype).to(device)
    model.eval()

    # Congelar 100% do modelo base
    for p in model.parameters():
        p.requires_grad = False
    print("Modelo base congelado com sucesso (apenas inferência de prefill).")

    # Inicializar a sonda
    probe = LatentHaltingProbe(hidden_size=config.hidden_size).to(device).to(torch.float32)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    cls_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    dataset = PromptProbeDataset(DATA_PATH, tokenizer, max_len=512)
    val_size = int(len(dataset) * 0.10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # Batch size 8 para velocidade máxima em GPU
    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True,
        collate_fn=lambda b: collate_prompt_batch(b, tokenizer)
    )
    val_loader = DataLoader(
        val_ds, batch_size=8, shuffle=False,
        collate_fn=lambda b: collate_prompt_batch(b, tokenizer)
    )

    epochs = 3
    print("\n" + "=" * 65)
    print(" INICIANDO TREINAMENTO ISOLADO DA SONDA LATENTE (3 ÉPOCAS)")
    print("=" * 65)

    t0 = time.time()
    for ep in range(epochs):
        probe.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_cls = batch["target_cls"].to(device)
            target_score = batch["target_score"].to(device)

            # Extrair estado intermediário do Loop 1 sem gradientes no transformer
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
                    h1 = out.loop_hidden_states[0]  # (B, S, H) - Exit 1 após camada 20

                    # Pegar a representação no último token válido do prompt (onde attention_mask == 1)
                    seq_lens = attention_mask.sum(dim=1) - 1
                    b_idx = torch.arange(h1.size(0), device=device)
                    h1_repr = h1[b_idx, seq_lens, :].float()

            # Passar pela sonda
            logits, score = probe(h1_repr)
            loss_cls = cls_criterion(logits, target_cls)
            loss_reg = reg_criterion(score, target_score)
            loss = loss_cls + (2.0 * loss_reg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=-1)
            correct += (pred == target_cls).sum().item()
            total += target_cls.size(0)

            if (step + 1) % 150 == 0 or (step + 1) == len(train_loader):
                acc = (correct / total) * 100
                avg_l = total_loss / (step + 1)
                print(f"Epoch {ep+1}/{epochs} | Passo [{step+1}/{len(train_loader)}] | "
                      f"Loss: {avg_l:.4f} | Acc Treino: {acc:.1f}%", flush=True)

        # Validação ao fim da época
        probe.eval()
        val_correct = 0
        val_total = 0
        val_mse = 0.0

        with torch.no_grad():
            for v_batch in val_loader:
                v_ids = v_batch["input_ids"].to(device)
                v_mask = v_batch["attention_mask"].to(device)
                v_cls = v_batch["target_cls"].to(device)
                v_score = v_batch["target_score"].to(device)

                with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                    v_out = model(input_ids=v_ids, attention_mask=v_mask)
                    v_h1 = v_out.loop_hidden_states[0]
                    v_lens = v_mask.sum(dim=1) - 1
                    v_repr = v_h1[torch.arange(v_h1.size(0), device=device), v_lens, :].float()

                v_logits, v_pred_score = probe(v_repr)
                v_pred = v_logits.argmax(dim=-1)
                val_correct += (v_pred == v_cls).sum().item()
                val_total += v_cls.size(0)
                val_mse += F.mse_loss(v_pred_score, v_score).item() * v_cls.size(0)

        val_acc = (val_correct / val_total) * 100
        val_avg_mse = val_mse / val_total
        print(f"\n--> [VALIDAÇÃO ÉPOCA {ep+1}] Acurácia: {val_acc:.1f}% | Erro Médio Contínuo (MSE): {val_avg_mse:.4f}\n", flush=True)

    t_tot = time.time() - t0
    print("=" * 65)
    print(" TREINAMENTO DA SONDA FINALIZADO COM SUCESSO!")
    print(f" Tempo total: {t_tot:.1f}s ({t_tot/60:.2f} min)")
    print(f" Acurácia Final de Roteamento: {val_acc:.1f}%")
    print(f" Salvando sonda em: {OUTPUT_PROBE_PATH}")
    print("=" * 65)

    torch.save(probe.state_dict(), OUTPUT_PROBE_PATH)
    print("Sonda latente salva e validada com sucesso!")

if __name__ == "__main__":
    train_probe()
