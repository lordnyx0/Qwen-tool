"""
train_dual_t4_grpo.py
Treinamento RIC-GRPO Especializado para Kaggle Dual GPU (2x NVIDIA Tesla T4 16GB = 32GB).
Executado via: torchrun --nproc_per_node=2 train_dual_t4_grpo.py --steps 100

Otimizações Ativas para Tesla T4:
1. Precisão: torch.float16 (Turing Tensor Cores nativos) com GradScaler
2. Distribuição DDP: 2 GPUs gerando G=3 cada (G=6 total por passo) em paralelo
3. Gradient Checkpointing nativo do Qwen3Loop em cada GPU (< 3.5 GB por placa)
4. Atenção SDPA (Memory-efficient attention)
5. Sincronização de vantagens e recompensas via torch.distributed
6. Calibração de Loop 1 via RMSNorm e Sonda Latente Halting Probe
"""

import os
import sys
import json
import random
import time
import math
from typing import List, Dict, Any, Tuple
from contextlib import nullcontext

# Configurar caminhos locais
sys.path.insert(0, os.path.abspath("engine"))
sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

import qwen3loop_python  # noqa: F401
from rewards_tool_calling import (
    reward_format,
    reward_tool_selection,
    reward_arguments,
    reward_abstention,
    reward_python_execution
)

# -------------------------------------------------------------
# Configurações Globais e Hiperparâmetros
# -------------------------------------------------------------
TOTAL_STEPS = 100            # 100 passos densos com G=6 (600 rollouts)
TOTAL_GROUP_SIZE = 6         # G=6 candidatos por prompt
MAX_NEW_TOKENS = 256         # Sem truncamento de qualidade
CLIP_EPS = 0.2
GAMMA_LOOP1 = 0.30
ALPHA_DISTILL = 0.10
LAMBDA_PROBE = 0.05
LR = 2e-5
PROBE_LR = 1e-4
EVAL_INTERVAL = 25
SAVE_INTERVAL = 25

class LatentHaltingProbe(nn.Module):
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

class DynamicCurriculumSampler:
    def __init__(self, data_dir: str):
        self.l1_data = self._load_jsonl(os.path.join(data_dir, "level_1_foundation.jsonl"))
        self.l2_data = self._load_jsonl(os.path.join(data_dir, "level_2_discrimination.jsonl"))
        self.l3_data = self._load_jsonl(os.path.join(data_dir, "level_3_reasoning_python.jsonl"))
        self.eval_data = self._load_jsonl(os.path.join(data_dir, "eval_benchmark_battery.jsonl"))

    def _load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        samples = []
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    item = json.loads(line)
                    tools = item.get("tools")
                    if isinstance(tools, str):
                        try: tools = json.loads(tools)
                        except Exception: tools = []
                    if isinstance(tools, dict): tools = [tools]
                    if not isinstance(tools, list): tools = []
                    item["tools"] = tools
                    samples.append(item)
        return samples

    def sample_prompt(self, current_step: int, total_steps: int) -> Dict[str, Any]:
        progress = current_step / max(1, total_steps)
        if progress < 0.30:
            pool = self.l1_data if random.random() < 0.70 else self.l2_data
        elif progress < 0.70:
            r = random.random()
            if r < 0.20: pool = self.l1_data
            elif r < 0.70: pool = self.l2_data
            else: pool = self.l3_data
        else:
            r = random.random()
            if r < 0.10: pool = self.l1_data
            elif r < 0.40: pool = self.l2_data
            else: pool = self.l3_data
        return random.choice(pool)

def run_evaluation_benchmark(model, tokenizer, eval_samples: List[Dict[str, Any]], device: torch.device, max_cases: int = 15) -> Dict[str, float]:
    model.eval()
    selected = eval_samples[:max_cases]
    total = len(selected)
    correct_format, correct_selection, correct_abstention, total_abstention = 0, 0, 0, 0
    
    with torch.no_grad():
        for case in selected:
            tools = case.get("tools")
            tools_arg = tools if (isinstance(tools, list) and len(tools) > 0) else None
            prompt_str = tokenizer.apply_chat_template(case["messages"], tools=tools_arg, add_generation_prompt=True, tokenize=False)
            inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=160, do_sample=False, repetition_penalty=1.05, pad_token_id=tokenizer.eos_token_id)
            gen_ids = outputs[0][inputs.input_ids.shape[1]:]
            comp = tokenizer.decode(gen_ids, skip_special_tokens=False).replace("<|im_end|>", "").strip()
            
            exp_call = case.get("expected_call")
            if reward_format([comp], expected_call=[exp_call])[0] > 0.5: correct_format += 1
            if reward_tool_selection([comp], expected_call=[exp_call])[0] > 0.5: correct_selection += 1
            if exp_call is None:
                total_abstention += 1
                if reward_abstention([comp], expected_call=[None])[0] > 0.5: correct_abstention += 1

    model.train()
    return {
        "eval_format_acc": (correct_format / total) * 100,
        "eval_selection_acc": (correct_selection / total) * 100,
        "eval_abstention_acc": (correct_abstention / max(1, total_abstention)) * 100
    }

def main():
    # Inicialização DDP
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_distributed:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Caminhos
    MODEL_DIR = os.environ.get("MODEL_DIR", "model")
    CURRICULUM_DIR = os.environ.get("CURRICULUM_DIR", "data")
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/kaggle/working/qwen3loop_tool_grpo_final")
    LOG_FILE = os.path.join(OUTPUT_DIR, "tool_grpo_training_log.json")

    # Local group size por GPU
    local_group_size = TOTAL_GROUP_SIZE // world_size

    if rank == 0:
        print("=" * 70, flush=True)
        print("INICIANDO TREINO RIC-GRPO EM DUAL GPU (KAGGLE 2x T4)", flush=True)
        print(f"GPUs Ativas:       {world_size}x Tesla T4 | Dtype: torch.float16", flush=True)
        print(f"Total Group Size:  G={TOTAL_GROUP_SIZE} ({local_group_size} por GPU)", flush=True)
        print(f"Passos Totais:     {TOTAL_STEPS} | Limite Geração: {MAX_NEW_TOKENS} tokens", flush=True)
        print("=" * 70, flush=True)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Carregar Tokenizer e Modelo
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config = AutoConfig.from_pretrained(MODEL_DIR)
    
    t_load0 = time.time()
    if rank == 0:
        print("Carregando pesos em bfloat16 nativo (mmap instantâneo)...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"
    )
    if rank == 0:
        print(f"Pesos lidos em {time.time() - t_load0:.2f}s. Movendo para GPU {rank} e convertendo para float16...", flush=True)
    base_model = base_model.to(device=device, dtype=torch.float16)
    if rank == 0:
        print(f"Modelo pronto na GPU em {time.time() - t_load0:.2f}s!", flush=True)

    base_model.gradient_checkpointing_enable()

    # 2. Configurar LoRA PEFT
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(base_model, peft_config)
    model.enable_input_require_grads()

    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    # 3. Sonda Latente
    probe = LatentHaltingProbe(config.hidden_size).to(device).to(torch.float16)
    probe_path = os.path.join(MODEL_DIR, "latent_halting_probe.pt")
    if os.path.exists(probe_path):
        try:
            probe.load_state_dict(torch.load(probe_path, map_location=device, weights_only=True))
            if rank == 0: print("Latent Halting Probe carregada com sucesso!", flush=True)
        except Exception:
            pass

    # 4. Otimizadores e Scaler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW([
        {"params": trainable_params, "lr": LR, "weight_decay": 0.01},
        {"params": probe.parameters(), "lr": PROBE_LR, "weight_decay": 0.01}
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=TOTAL_STEPS, eta_min=1e-6)
    scaler = GradScaler()

    # 5. Currículo
    curriculum = DynamicCurriculumSampler(CURRICULUM_DIR)
    training_history = []
    step_times = []
    start_time = time.time()

    if rank == 0:
        print("\n--- TREINAMENTO ATIVO INICIADO NO KAGGLE ---\n", flush=True)

    for step in range(1, TOTAL_STEPS + 1):
        step_t0 = time.time()
        torch.cuda.reset_peak_memory_stats(device)

        # Sincronizar o mesmo seed por passo para que ambos os ranks usem o mesmo prompt
        random.seed(42 + step)
        sample = curriculum.sample_prompt(step, TOTAL_STEPS)

        tools = sample.get("tools")
        tools_arg = tools if (isinstance(tools, list) and len(tools) > 0) else None
        prompt_str = tokenizer.apply_chat_template(sample["messages"], tools=tools_arg, add_generation_prompt=True, tokenize=False)
        prompt_inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
        prompt_len = prompt_inputs.input_ids.shape[1]

        # Resetar seed local para exploração independente de cada GPU
        torch.manual_seed(1000 * (rank + 1) + step)

        # A. Rollout Paralelo: Cada GPU gera local_group_size respostas simultaneamente
        unwrapped_model = model.module if is_distributed else model
        unwrapped_model.eval()
        with torch.no_grad():
            rollout_outputs = unwrapped_model.generate(
                **prompt_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=local_group_size,
                pad_token_id=tokenizer.eos_token_id
            )
        unwrapped_model.train()

        # Decodificar textos locais
        local_completions = []
        for i in range(local_group_size):
            gen_tokens = rollout_outputs[i][prompt_len:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=False).replace("<|im_end|>", "").strip()
            local_completions.append(text)

        # B. Coletar todas as G=6 respostas entre as GPUs para calcular recompensas e vantagens globais
        if is_distributed:
            gathered_completions = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_completions, local_completions)
            all_completions = []
            for comp_list in gathered_completions:
                all_completions.extend(comp_list)
        else:
            all_completions = local_completions

        exp_call = sample.get("expected_call")
        exp_res = sample.get("expected_execution_result")

        # Avaliar recompensas hierárquicas (HTGR)
        r_fmt = reward_format(all_completions, expected_call=[exp_call]*TOTAL_GROUP_SIZE)
        r_sel = reward_tool_selection(all_completions, expected_call=[exp_call]*TOTAL_GROUP_SIZE)
        r_arg = reward_arguments(all_completions, expected_call=[exp_call]*TOTAL_GROUP_SIZE)
        r_abs = reward_abstention(all_completions, expected_call=[exp_call]*TOTAL_GROUP_SIZE)
        r_py  = reward_python_execution(all_completions, expected_execution_result=[exp_res]*TOTAL_GROUP_SIZE)

        all_rewards = [
            (rf * 1.0) + (rs * 1.0) + (ra * 1.0) + (rab * 1.5) + (rpy * 2.0)
            for rf, rs, ra, rab, rpy in zip(r_fmt, r_sel, r_arg, r_abs, r_py)
        ]

        mean_r = sum(all_rewards) / TOTAL_GROUP_SIZE
        var_r = sum((r - mean_r) ** 2 for r in all_rewards) / TOTAL_GROUP_SIZE
        std_r = math.sqrt(var_r) + 1e-4
        all_advantages = [(r - mean_r) / std_r for r in all_rewards]

        # Fatiar vantagens locais correspondentes a este rank
        local_advantages = all_advantages[rank * local_group_size : (rank + 1) * local_group_size]

        # C. Forward & Backward Pass Vetorizado
        step_ppl_l2 = 0.0
        step_ppl_l1 = 0.0
        optimizer.zero_grad()

        for i in range(local_group_size):
            seq_ids = rollout_outputs[i].unsqueeze(0)
            adv = local_advantages[i]

            sync_context = model.no_sync() if (is_distributed and i < local_group_size - 1) else nullcontext()
            with sync_context:
                with autocast(dtype=torch.float16):
                    outputs = model(input_ids=seq_ids, output_hidden_states=False, return_dict=True)

                    logits_l2 = outputs.logits[:, prompt_len - 1 : -1, :]
                    targets = seq_ids[:, prompt_len:]

                    log_probs_l2 = F.log_softmax(logits_l2, dim=-1)
                    token_log_probs_l2 = torch.gather(log_probs_l2, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
                    nll_l2 = -token_log_probs_l2.mean()
                    step_ppl_l2 += torch.exp(nll_l2.detach()).item()
                    loss_policy_l2 = nll_l2 * adv

                    # Loop 1 com RMSNorm
                    loop_hidden = getattr(outputs, "loop_hidden_states", None)
                    loss_policy_l1 = torch.tensor(0.0, device=device)
                    loss_distill = torch.tensor(0.0, device=device)

                    if loop_hidden and len(loop_hidden) > 1:
                        raw_model = unwrapped_model.base_model.model
                        h_l1_raw = loop_hidden[len(loop_hidden) // 2]
                        h_l1 = raw_model.norm(h_l1_raw)
                        logits_l1 = unwrapped_model.base_model.lm_head(h_l1[:, prompt_len - 1 : -1, :])
                        log_probs_l1 = F.log_softmax(logits_l1, dim=-1)
                        token_log_probs_l1 = torch.gather(log_probs_l1, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
                        nll_l1 = -token_log_probs_l1.mean()
                        step_ppl_l1 += torch.exp(nll_l1.detach()).item()
                        loss_policy_l1 = nll_l1 * adv

                        probs_l2_detach = F.softmax(logits_l2.detach(), dim=-1)
                        loss_distill = F.kl_div(log_probs_l1, probs_l2_detach, reduction="batchmean")

                        # Sonda
                        h_prompt_end = h_l1[:, prompt_len - 1, :]
                        probe_logits, probe_score = probe(h_prompt_end)
                        if exp_call is None: target_cls = torch.tensor([0], device=device)
                        elif "python" in exp_call.get("name", ""): target_cls = torch.tensor([2], device=device)
                        else: target_cls = torch.tensor([1], device=device)
                        loss_probe = F.cross_entropy(probe_logits, target_cls)
                    else:
                        loss_probe = torch.tensor(0.0, device=device)

                    total_sample_loss = (
                        loss_policy_l2 +
                        (GAMMA_LOOP1 * loss_policy_l1) +
                        (ALPHA_DISTILL * loss_distill) +
                        (LAMBDA_PROBE * loss_probe)
                    ) / (local_group_size * world_size)

                scaler.scale(total_sample_loss).backward()
                del outputs, total_sample_loss

        # D. Passo de Otimização com GradScaler
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params + list(probe.parameters()), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # E. Logs no Rank 0
        step_elapsed = time.time() - step_t0
        step_times.append(step_elapsed)
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024**3)

        avg_step_ppl_l2 = step_ppl_l2 / local_group_size
        avg_step_ppl_l1 = (step_ppl_l1 / local_group_size) if step_ppl_l1 > 0 else avg_step_ppl_l2
        phase = 1 if (step / TOTAL_STEPS) < 0.30 else (2 if (step / TOTAL_STEPS) < 0.70 else 3)

        avg_time = sum(step_times[-10:]) / len(step_times[-10:])
        eta_minutes = ((TOTAL_STEPS - step) * avg_time) / 60

        if rank == 0:
            print(
                f"[Step {step:03d}/{TOTAL_STEPS}] Fase {phase} | "
                f"Reward: {mean_r:.3f} | "
                f"PPL-L2: {avg_step_ppl_l2:.2f} | "
                f"PPL-L1: {avg_step_ppl_l1:.2f} | "
                f"VRAM T4: {peak_vram_gb:.2f} GB | "
                f"Tempo: {step_elapsed:.1f}s | "
                f"ETA: {eta_minutes:.1f} min",
                flush=True
            )
            training_history.append({
                "step": step, "phase": phase, "reward": mean_r,
                "ppl_loop2": avg_step_ppl_l2, "ppl_loop1": avg_step_ppl_l1,
                "peak_vram_gb": peak_vram_gb, "step_time_sec": step_elapsed, "eta_min": eta_minutes
            })
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(training_history, f, indent=2)

        # F. Checkpoints Periódicos
        if step % SAVE_INTERVAL == 0 and rank == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped_model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(probe.state_dict(), os.path.join(ckpt_dir, "latent_halting_probe.pt"))
            print(f"Checkpoint intermediário salvo em: {ckpt_dir}\n", flush=True)

    # 6. Finalização
    if rank == 0:
        total_time = (time.time() - start_time) / 60
        print("\n" + "=" * 70, flush=True)
        print(f"TREINAMENTO RIC-GRPO CONCLUÍDO NO KAGGLE EM {total_time:.2f} MINUTOS!", flush=True)
        unwrapped_model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        torch.save(probe.state_dict(), os.path.join(OUTPUT_DIR, "latent_halting_probe.pt"))
        print(f"Modelo final salvo em: {OUTPUT_DIR}", flush=True)
        print("=" * 70, flush=True)

    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
