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
import re
import shutil
import sys
import json
import math
import time
import argparse
from typing import Dict, List, Optional, Tuple

# Resolução de caminhos da engine qwen3loop_python (prioridade absoluta para /kaggle/working)
cur_dir = os.path.dirname(os.path.abspath(__file__))
engine_search = [
    "/kaggle/working/engine",
    "/kaggle/working",
    cur_dir,
    os.path.abspath(os.path.join(cur_dir, "engine")),
    os.path.abspath(os.path.join(cur_dir, "..", "engine")),
    os.path.abspath("engine"),
]
for p in engine_search:
    p_abs = os.path.abspath(p)
    if os.path.exists(p_abs) and p_abs not in sys.path:
        sys.path.insert(0, p_abs)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import subprocess
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import Accelerator, DistributedDataParallelKwargs

try:
    import qwen3loop_python
    from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM
    AutoConfig.register("qwen3loop", Qwen3LoopConfig)
    AutoModelForCausalLM.register(Qwen3LoopConfig, Qwen3LoopForCausalLM)
    print(f"Engine qwen3loop_python carregada com sucesso de: {getattr(qwen3loop_python, '__file__', 'local')}")
except Exception as e:
    print(f"Aviso ao carregar qwen3loop_python: {e}")

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


_ERR_RE = re.compile(
    r"ENOENT|EISDIR|Traceback|PermissionError|FileNotFoundError|"
    r"NotJSONError|Invalid|failed|ERROR|Error:|does not exist|"
    r"not found|No such|aborted")


def repeat_violation_mask(messages, tools):
    """Indices (0-based entre turnos assistant) a mascarar no loss (C1/C1b).

    C1: turno cuja call repete (nome+args) call anterior que foi seguida de
    erro (não ensinar retry idêntico como alvo).
    C1b: turno cuja call viola schema visível (args fora de properties
    não-vazio). Schemas vazios não restringem (sem informação).
    """
    turns = [m for m in (messages or []) if m.get("role") == "assistant"]
    schemas = {}
    for t in (tools or []):
        fn = t.get("function", {})
        schemas[fn.get("name", "")] = set(
            ((fn.get("parameters", {}) or {}).get("properties", {}) or {}).keys())
    masked = set()
    seen = {}
    for j, m in enumerate(turns):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            key = (fn.get("name", ""), fn.get("arguments", ""))
            if key in seen:
                # C1: a primeira ocorrência foi seguida de erro?
                first_idx = seen[key]
                err_after = False
                count = 0
                in_first = False
                for gm in (messages or []):
                    if gm.get("role") == "assistant":
                        if count == first_idx:
                            in_first = True
                        count += 1
                    elif gm.get("role") == "tool" and in_first:
                        if _ERR_RE.search(gm.get("content", "")):
                            err_after = True
                        break
                if err_after:
                    masked.add(j)
            else:
                seen[key] = j
            # C1b: viola schema visível?
            sch = schemas.get(fn.get("name", ""), set())
            if sch:
                try:
                    a = json.loads(fn.get("arguments", "{}"))
                except Exception:
                    a = {}
                if isinstance(a, dict) and any(k not in sch for k in a.keys()):
                    masked.add(j)
    return masked


CURATED_ERR_OK = {"manual_v35", "manual_v37_alias"}


def error_follow_mask(messages, source):
    """Indices (0-based entre turnos assistant) a mascarar no loss (C1c).

    C1c: turno com call seguida de tool-output de erro (Dataset B sem
    dividir arquivos: nao treinar a tentativa errada como alvo, treinar a
    correcao). Excecao: fontes curadas error-first (P4 e aliases), cuja
    pedagogia e tentar-ler-erro-trocar.
    """
    if source in CURATED_ERR_OK:
        return set()
    turns = [m for m in (messages or []) if m.get("role") == "assistant"]
    if not any(m.get("tool_calls") for m in turns):
        return set()
    masked = set()
    j = -1
    i = 0
    msgs = messages or []
    while i < len(msgs):
        m = msgs[i]
        if m.get("role") == "assistant":
            j += 1
            if m.get("tool_calls"):
                k = i + 1
                while k < len(msgs) and (msgs[k].get("role") not in ("assistant", "user")):
                    gm = msgs[k]
                    if gm.get("role") == "tool" and _ERR_RE.search(gm.get("content", "")):
                        masked.add(j)
                        break
                    k += 1
        i += 1
    return masked


def false_irrelevance_mask(messages, uuid, uuid_set):
    """Indices (0-based entre turnos assistant) a mascarar no loss (C3).

    C3: primeiro turno textual do assistente em uuids listados (falsa
    irrelevância herdada da fonte: negação/resposta-de-memória antes da
    call, com meio carregando info real — colapso inseguro). Treina a
    call, não a negação.
    """
    if not uuid_set or uuid not in uuid_set:
        return set()
    turns = [m for m in (messages or []) if m.get("role") == "assistant"]
    for j, m in enumerate(turns):
        if not m.get("tool_calls"):
            return {j}
    return set()


class Qwen3LoopToolDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_len: int = 1024,
                 mask_repeats: bool = True, mask_first_file: str = ""):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_repeats = mask_repeats
        self.mask_first_set = set()
        if mask_first_file:
            try:
                with open(mask_first_file, "r", encoding="utf-8") as f:
                    self.mask_first_set = set(json.load(f).get("uuids", []))
                print(f"Mascarando 1o turno textual (C3) em {len(self.mask_first_set)} uuids.")
            except Exception as e:
                print(f"AVISO: mask_first_file ilegivel ({e}); C3 desligada.")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_repeats = mask_repeats
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

        # Defesa ativa contra chaves inválidas em tool_calls (ex: parameters em vez de arguments)
        for m in msgs:
            if isinstance(m, dict) and "tool_calls" in m and m["tool_calls"]:
                for tc in m["tool_calls"]:
                    if isinstance(tc, dict):
                        f = tc.get("function", {})
                        if isinstance(f, dict):
                            if "arguments" not in f and "parameters" in f:
                                f["arguments"] = f.pop("parameters")
                            if "arguments" not in f:
                                f["arguments"] = "{}"
                            if isinstance(f.get("arguments"), dict):
                                f["arguments"] = json.dumps(f["arguments"])

        # Gerar tokens da conversa completa via apply_chat_template nativo
        try:
            full_txt = self.tokenizer.apply_chat_template(
                msgs,
                tools=tools,
                add_generation_prompt=False,
                tokenize=False
            )
        except Exception:
            return self.__getitem__(0)

        full_ids = self.tokenizer.encode(full_txt, add_special_tokens=False)

        # Descartar estritamente se exceder max_len para evitar respostas incompletas sem <|im_end|>
        if len(full_ids) > self.max_len:
            return self.__getitem__((idx + 1) % len(self.samples))

        # assistant_only_loss: loss apenas nos turnos do assistente (textuais e tool_calls)
        # system, user e tool (observações) são mascarados com -100.
        # C1/C1b (v35): turnos que repetem call idêntica pós-erro ou que violam
        # schema visível também são mascarados (não ensinar o erro como alvo).
        masked_turns = repeat_violation_mask(msgs, tools) if self.mask_repeats else set()
        masked_turns |= false_irrelevance_mask(msgs, row.get("uuid", ""), self.mask_first_set)
        if self.mask_repeats:
            masked_turns |= error_follow_mask(msgs, row.get("source", ""))
        labels = [-100] * len(full_ids)
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_id = self.tokenizer.convert_tokens_to_ids("assistant")
        nl_id = self.tokenizer.convert_tokens_to_ids("\n")

        i = 0
        turn_idx = -1
        while i < len(full_ids):
            # Detectar início do turno do assistente: <|im_start|> assistant
            if full_ids[i] == im_start_id and i + 1 < len(full_ids) and full_ids[i+1] == assistant_id:
                turn_idx += 1
                i += 2
                if i < len(full_ids) and (full_ids[i] == nl_id or full_ids[i] == 198):
                    i += 1
                # Turno ativo do assistente: calcula loss até o <|im_end|>
                # (turnos em masked_turns (C1/C1b) ficam em -100)
                masked = turn_idx in masked_turns
                while i < len(full_ids):
                    if full_ids[i] == im_end_id:
                        if not masked:
                            labels[i] = full_ids[i]  # Inclui <|im_end|> para aprender a encerrar
                        i += 1
                        break
                    else:
                        if not masked:
                            labels[i] = full_ids[i]
                        i += 1
            else:
                i += 1

        # Alvo da sonda de recorrência:
        if "difficulty" in row:
            diff = row["difficulty"]
            target_rec = max(0, min(2, diff - 1))
        elif row.get("is_retrajectory"):
            target_rec = 2  # Retrajetórias exigem raciocínio corretivo profundo
        elif row.get("pillar"):
            pillar = row.get("pillar")
            if pillar in ["openclaw_coder", "powershell_windows"]:
                target_rec = 0 if row.get("token_length", 0) < 150 else 1
            elif pillar in ["toolmind_graphsyn", "synth_memory"]:
                target_rec = 1
            elif pillar in ["nemotron_terminal", "hermes_agent_traces"]:
                target_rec = 2
            else:
                target_rec = 1
        else:
            cat = row.get("category", "")
            if cat in ["negative_abstention", "single_tool_precise", "policy_compliance_and_clarification"]:
                target_rec = 0  # 1 loop (Early Exit / Clarificação rápida)
            elif cat in ["multi_tool_discrimination", "single_turn_tool_call"]:
                target_rec = 1  # 2 loops (Standard)
            elif cat in ["python_algorithmic_reasoning", "multi_turn_agentic_chaining"]:
                target_rec = 2  # 3+ loops (Deep Recurrence / Chaining)
            else:
                target_rec = 1

        # Fallback defensivo: se não houver tokens ativos, avança para a próxima amostra válida
        if not any(x != -100 for x in labels):
            return self.__getitem__((idx + 1) % len(self.samples))

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
    mixed_prec = args.mixed_precision
    if mixed_prec == "bf16" and not torch.cuda.is_bf16_supported():
        mixed_prec = "fp16"

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=mixed_prec,
        kwargs_handlers=[ddp_kwargs]
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print(f"Dispositivo de treino principal: {device} | Total de processos DDP: {accelerator.num_processes}")
        print(f"Precisão mista DDP (Accelerate): {accelerator.mixed_precision}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} | VRAM Total: {props.total_memory / 1024**3:.2f} GB")

    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        major_cc = torch.cuda.get_device_capability()[0]
        if major_cc < 8:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if accelerator.is_main_process:
            print("Backends SDPA: Flash=False | Mem-Efficient=True | Math=True (Otimizado para Turing T4)")
    else:
        dtype = torch.float32
    if accelerator.is_main_process:
        print(f"Precisão de ativações e pesos: {dtype}")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # 2. Config & Modelo Looped (supervisão profunda permanente: nunca vazio)
    loop_loss_weights = [float(w.strip()) for w in args.loop_loss_weights.split(",") if w.strip()] if args.loop_loss_weights else []
    assert len(loop_loss_weights) > 0, "loop_loss_weights vazio: treino sem supervisão intermediária"
    if accelerator.is_main_process:
        print(f"Configurando Qwen3Loop: num_loops={args.num_loops}, loop_loss_weights={loop_loss_weights}")

    config = AutoConfig.from_pretrained(
        args.model_path,
        num_loops=args.num_loops,
        loop_loss_weights=loop_loss_weights
    )
    config._attn_implementation = "sdpa"
    if accelerator.is_main_process:
        print(f"Interface de atenção explicitamente configurada: {config._attn_implementation}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=dtype,
        attn_implementation="sdpa"
    )

    # Ativar Gradient Checkpointing
    model.gradient_checkpointing_enable()
    if accelerator.is_main_process:
        print("Gradient Checkpointing ativado com sucesso.")

    # 3. Sonda Latente de Recorrência
    probe_head = None
    if args.train_probe:
        probe_head = LatentHaltingProbe(hidden_size=config.hidden_size, num_classes=3).to(dtype)
        if accelerator.is_main_process:
            print("Sonda de Halting Latente inicializada.")

    # 4. Dataset & Split Treino / Val (val_frac=0 usa 100% p/ treino)
    full_dataset = Qwen3LoopToolDataset(args.data_path, tokenizer, max_len=args.max_len,
                                        mask_repeats=args.mask_repeats,
                                        mask_first_file=args.mask_first_turn_file)
    if args.val_frac and args.val_frac > 0:
        val_size = max(20, int(len(full_dataset) * args.val_frac))
        train_size = len(full_dataset) - val_size
        train_ds, val_ds = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
    else:
        train_ds, val_ds = full_dataset, None
        train_size, val_size = len(full_dataset), 0

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
    ) if val_ds is not None else None

    if accelerator.is_main_process:
        print(f"Divisão: {train_size:,} treino | {val_size:,} validação")

    # 5. Otimizador (AdamW 8-bit ou Fallback)
    params = list(model.parameters())
    if probe_head is not None:
        params += list(probe_head.parameters())

    if HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.01)
        if accelerator.is_main_process:
            print("Otimizador: BitsAndBytes AdamW 8-bit")
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
        if accelerator.is_main_process:
            print("Otimizador: PyTorch AdamW standard")

    probe_criterion = nn.CrossEntropyLoss()

    # Preparação DDP com Accelerator
    if probe_head is not None:
        model, probe_head, optimizer, train_loader, val_loader = accelerator.prepare(
            model, probe_head, optimizer, train_loader, val_loader
        )
    else:
        model, optimizer, train_loader, val_loader = accelerator.prepare(
            model, optimizer, train_loader, val_loader
        )

    effective_batch = args.batch_size * args.grad_accum * accelerator.num_processes
    total_steps = (len(train_loader) * args.epochs) // args.grad_accum
    if args.max_steps is not None and args.max_steps < total_steps:
        total_steps = args.max_steps

    if accelerator.is_main_process:
        print(f"Total de passos de otimizador: {total_steps:,} (Batch efetivo global: {effective_batch})")
        os.makedirs(args.output_dir, exist_ok=True)

    # 6. Loop de Treino
    model.train()
    if probe_head:
        probe_head.train()

    step = 0
    running_lm_loss = 0.0
    best_dev = -1.0
    best_epoch = 0
    dev_wait = 0
    running_probe_loss = 0.0
    probe_correct = 0
    probe_total = 0
    t0 = time.time()

    optimizer.zero_grad()

    if accelerator.is_main_process:
        print("\n" + "=" * 65)
        print(" INICIANDO TREINAMENTO SFT TOOL-CALLING COM MULTI-EXIT LOSS (DDP)")
        print("=" * 65)

    for epoch in range(args.epochs):
        for batch in train_loader:
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                labels = batch["labels"]
                attention_mask = batch["attention_mask"]
                target_rec = batch["target_rec"]

                with torch.autocast(device_type=device.type, dtype=dtype, enabled=torch.cuda.is_available()):
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
                        last_idx = attention_mask.sum(dim=-1) - 1
                        h1_repr = h1[torch.arange(h1.size(0), device=h1.device), last_idx]
                        probe_logits = probe_head(h1_repr)
                        p_loss = probe_criterion(probe_logits, target_rec)
                        loss = loss + (0.15 * p_loss)

                        pred = probe_logits.argmax(dim=-1)
                        probe_correct += (pred == target_rec).sum().item()
                        probe_total += target_rec.size(0)

                    # Fail-fast permanente: sem 2+ saídas, não há supervisão profunda
                    # (flag, não step==0: step 0 dura grad_accum micro-batches)
                    if not getattr(accelerator, "_exits_checked", False):
                        n_exits = len(getattr(outputs, "loop_hidden_states", None) or [])
                        assert n_exits > 1, f"supervisão intermediária inativa (exits={n_exits})"
                        if accelerator.is_main_process:
                            print(f"Supervisão profunda ativa: {n_exits} saídas intermediárias + final", flush=True)
                        accelerator._exits_checked = True

                accelerator.backward(loss)

                running_lm_loss += (lm_loss.item() / args.grad_accum)
                running_probe_loss += (p_loss.item() / args.grad_accum)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params, max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1

                    # Log periódico a cada log_steps no rank 0 com cálculo de ETA
                    if step % args.log_steps == 0 and accelerator.is_main_process:
                        avg_lm = running_lm_loss / args.log_steps
                        avg_probe = running_probe_loss / args.log_steps
                        probe_acc = (probe_correct / max(1, probe_total)) * 100 if probe_total > 0 else 0
                        elapsed = time.time() - t0
                        vram_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if torch.cuda.is_available() else 0
                        speed = step / max(1.0, elapsed)
                        eta_seconds = (total_steps - step) / max(1e-4, speed)
                        eta_m, eta_s = divmod(int(eta_seconds), 60)
                        eta_h, eta_m = divmod(eta_m, 60)
                        eta_str = f"{eta_h:02d}h{eta_m:02d}m" if eta_h > 0 else f"{eta_m:02d}m{eta_s:02d}s"

                        print(f"Passo [{step:,}/{total_steps:,}] ({step/max(1, total_steps)*100:.1f}%) | Epoch {epoch+1} | "
                              f"LM Loss: {avg_lm:.4f} | Probe Loss: {avg_probe:.4f} | "
                              f"Probe Acc: {probe_acc:.1f}% | VRAM: {vram_gb:.2f} GB | "
                              f"Tempo: {elapsed:.1f}s | ETA: {eta_str}", flush=True)

                        running_lm_loss = 0.0
                        running_probe_loss = 0.0
                        probe_correct = 0
                        probe_total = 0

                    # Avaliação de validação (só se houver val_loader)
                    if val_loader is not None and step % args.eval_steps == 0:
                        evaluate(model, probe_head, val_loader, accelerator, dtype, probe_criterion)

                    # Salvamento de checkpoint (com rotacao p/ teto de disco)
                    if step % args.save_steps == 0 and accelerator.is_main_process:
                        ckpt_path = os.path.join(args.output_dir, f"checkpoint-{step}")
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_probe = accelerator.unwrap_model(probe_head) if probe_head else None
                        save_model_and_probe(unwrapped_model, unwrapped_probe, tokenizer, config, ckpt_path)
                        rotate_step_checkpoints(args.output_dir, keep=args.save_total_limit)

                    if args.max_steps is not None and step >= args.max_steps:
                        break

        # Fim de época: eval dev-hard + ckpt + early-stop (só época completa)
        epoch_done = not (args.max_steps is not None and step >= args.max_steps)
        if epoch_done and args.dev_eval and accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            dev = evaluate_dev(unwrapped, tokenizer, args.dev_eval)
            print(f"\n[ÉPOCA {epoch+1}] dev-hard: {dev['correct']}/{dev['total']} "
                  f"({dev['dev_overall']*100:.1f}%) por cat: {dev['by_cat']}", flush=True)
            ckpt_epoch = os.path.join(args.output_dir, f"epoch-{epoch+1}")
            save_model_and_probe(
                unwrapped,
                accelerator.unwrap_model(probe_head) if probe_head else None,
                tokenizer, config, ckpt_epoch)
            if dev["dev_overall"] > best_dev:
                best_dev = dev["dev_overall"]
                best_epoch = epoch + 1
                dev_wait = 0
            else:
                dev_wait += 1
            print(f"Melhor: época {best_epoch} ({best_dev*100:.1f}%) | "
                  f"paciência: {dev_wait}/{args.patience}", flush=True)
            if dev_wait >= args.patience:
                print(f"EARLY-STOP: {args.patience} época(s) sem melhora. "
                      f"Melhor checkpoint: epoch-{best_epoch}", flush=True)
                break

        if args.max_steps is not None and step >= args.max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        print("\n" + "=" * 65)
        print(" TREINAMENTO FINALIZADO!")
        print(f" Salvando modelo final em: {args.output_dir}")
        print("=" * 65)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_probe = accelerator.unwrap_model(probe_head) if probe_head else None
        save_model_and_probe(unwrapped_model, unwrapped_probe, tokenizer, config, args.output_dir)


def evaluate(model, probe_head, val_loader, accelerator, dtype, probe_criterion, max_batches: Optional[int] = 30):
    model.eval()
    if probe_head:
        probe_head.eval()

    total_lm_loss = 0.0
    correct = 0
    total = 0
    eval_count = 0
    device = accelerator.device

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]
            target_rec = batch["target_rec"]

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_lm_loss += outputs.loss.item()

                if probe_head is not None and hasattr(outputs, "loop_hidden_states") and len(outputs.loop_hidden_states) > 0:
                    h1 = outputs.loop_hidden_states[0]
                    last_idx = attention_mask.sum(dim=-1) - 1
                    h1_repr = h1[torch.arange(h1.size(0), device=h1.device), last_idx]
                    probe_logits = probe_head(h1_repr)
                    pred = probe_logits.argmax(dim=-1)
                    correct += (pred == target_rec).sum().item()
                    total += target_rec.size(0)
            eval_count += 1

    avg_loss = total_lm_loss / max(1, eval_count)
    ppl = math.exp(min(20, avg_loss))
    acc = (correct / max(1, total)) * 100 if total > 0 else 0
    if accelerator.is_main_process:
        print(f"\n[VALIDAÇÃO ({eval_count} batches)] Val Loss: {avg_loss:.4f} | Perplexidade: {ppl:.2f} | Probe Acc: {acc:.1f}%\n", flush=True)

    model.train()
    if probe_head:
        probe_head.train()


def rotate_step_checkpoints(output_dir, keep=2):
    """Apaga dirs checkpoint-{step} antigos, mantendo os `keep` mais recentes.

    Teto de disco p/ Kaggle: cada ckpt full tem ~1.2GB e sem rotacao 3 epocas
    x 14 saves estouram o /kaggle/working (v33 morreu com ENOSPC no save final).
    Nunca toca em epoch-{n} (early-stop) nem nos arquivos da raiz.
    """
    if keep is not None and keep < 1:
        return
    try:
        entries = []
        for name in os.listdir(output_dir):
            m = re.fullmatch(r"checkpoint-(\d+)", name)
            if m and os.path.isdir(os.path.join(output_dir, name)):
                entries.append((int(m.group(1)), name))
    except FileNotFoundError:
        return
    entries.sort()
    for _, name in entries[:-keep] if keep is not None else []:
        path = os.path.join(output_dir, name)
        try:
            size_gb = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(path) for f in fs) / 1e9
            shutil.rmtree(path)
            print(f"Rotacao: removido {name} ({size_gb:.2f} GB liberados)", flush=True)
        except Exception as e:
            print(f"Rotacao: falha ao remover {name}: {e}", flush=True)


def save_model_and_probe(model, probe_head, tokenizer, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Salvando pesos e configurações em {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    if probe_head is not None:
        torch.save(probe_head.state_dict(), os.path.join(save_dir, "latent_halting_probe.pt"))
    print("Checkpoint salvo com sucesso!")


def evaluate_dev(model, tokenizer, dev_path, max_new_tokens=256):
    """Avalia geração greedy no dev-hard (formato bateria) p/ seleção de época.

    Usa as mesmas funções de reward do treino como scorers + execução real
    de Python. Retorna overall (0..1) e acertos por categoria.
    """
    import sys as _sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from rewards_tool_calling import (
        extract_tool_calls_from_text, reward_format, reward_tool_selection,
        reward_arguments,
    )
    items = [json.loads(l) for l in open(dev_path, encoding="utf-8") if l.strip()]
    device = next(model.parameters()).device
    model.eval()
    correct = total = 0
    cat_hit, cat_tot = {}, {}
    with torch.no_grad():
        for it in items:
            tools = it.get("tools")
            tools_arg = tools if (isinstance(tools, list) and len(tools) > 0) else None
            prompt = tokenizer.apply_chat_template(
                it["messages"], tools=tools_arg, add_generation_prompt=True, tokenize=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id)
            comp = tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=False).replace("<|im_end|>", "").strip()
            exp_tool = it.get("expected_tool") or None
            exp_args = it.get("expected_args") or {}
            if isinstance(exp_args, str):
                try:
                    exp_args = json.loads(exp_args)
                except Exception:
                    exp_args = {}
            cat = it.get("category", "?")
            cat_tot[cat] = cat_tot.get(cat, 0) + 1
            total += 1
            ok = False
            if not exp_tool:
                calls, _ = extract_tool_calls_from_text(comp)
                ok = (len(calls) == 0)
            elif exp_tool in ("execute_python", "python_interpreter") or it.get("ground_truth_stdout"):
                calls, _ = extract_tool_calls_from_text(comp)
                py = [c for c in calls if c.get("name") in ("execute_python", "python_interpreter")]
                if py:
                    code = (py[0].get("arguments") or {}).get("code", "")
                    try:
                        r = subprocess.run(
                            [_sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=10)
                        outs = r.stdout.strip()
                        exps = str(it.get("ground_truth_stdout", "")).strip()
                        ok = (outs == exps) or (exps in outs)
                    except Exception:
                        ok = False
            else:
                exp_call = {"name": exp_tool, "arguments": exp_args}
                rf = reward_format([comp], expected_call=[exp_call])[0]
                rs = reward_tool_selection([comp], expected_call=[exp_call])[0]
                ra = reward_arguments([comp], expected_call=[exp_call])[0]
                ok = (rf >= 1.0 and rs >= 1.0 and ra >= 1.0)
            if ok:
                correct += 1
                cat_hit[cat] = cat_hit.get(cat, 0) + 1
    model.train()
    return {"dev_overall": correct / max(1, total), "correct": correct, "total": total,
            "by_cat": {k: (cat_hit.get(k, 0), v) for k, v in cat_tot.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino SFT Tool-Calling Qwen3Loop")
    parser.add_argument("--data_path", type=str, default="curriculum_3000_tool_sft.jsonl")
    parser.add_argument("--model_path", type=str, default="checkpoints/qwen3loop_tool_sft_final")
    parser.add_argument("--output_dir", type=str, default="checkpoints/qwen3loop_tool_curriculum_3000_final")
    parser.add_argument("--num_loops", type=int, default=2)
    parser.add_argument("--loop_loss_weights", type=str, default="0.4,1.0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--max_len", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--patience", type=int, default=1,
                        help="Épocas sem melhora no dev-hard antes de parar (requer --dev_eval)")
    parser.add_argument("--dev_eval", type=str, default=None,
                        help="JSONL dev (formato bateria) avaliado ao fim de cada época p/ seleção/early-stop")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--eval_steps", type=int, default=25)
    parser.add_argument("--val_frac", type=float, default=0.05,
                        help="Fracao do dataset reservada p/ validacao (0.0 = treina em 100%%)")
    parser.add_argument("--save_steps", type=int, default=35)
    parser.add_argument("--mask_repeats", action=argparse.BooleanOptionalAction, default=True,
                        help="Mascara turnos que repetem call identica pos-erro ou violam schema (C1/C1b)")
    parser.add_argument("--mask_first_turn_file", type=str, default="",
                        help="JSON {uuids:[...]} p/ mascarar o 1o turno textual (C3 falsa-irrelevancia); vazio = desligada")
    parser.add_argument("--save_total_limit", type=int, default=2,
                        help="Max de dirs checkpoint-{step} mantidos (rotacao p/ teto de disco; epoch-{n} preservados)")
    parser.add_argument("--log_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    parser.add_argument("--train_probe", action="store_true", default=True)

if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
