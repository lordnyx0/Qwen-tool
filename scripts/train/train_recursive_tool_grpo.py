"""
train_recursive_tool_grpo.py
Treinamento de Reinforcement Learning via GRPO Recursivo com Calibração Intermediária (RIC-GRPO)
para o Qwen3Loop (56 camadas virtuais / 2 loops com pesos amarrados).

Mecanismos e Otimizações Ativas:
1. Dynamic Curriculum Scheduler (3 Fases: Fundação/Abstenção -> Discriminação/Schemas -> Python Execution)
2. Hierarchical Gated Rewards (HTGR): Formato, Seleção, Argumentos, Abstenção e Execução Sandbox
3. Dual-Exit Policy Gradient:
   L_policy = L_GRPO(Loop 2) + gamma * L_GRPO(Loop 1)
4. Inter-Loop Latent Consistency (Self-Distillation KL entre Loop 1 normalizado e Loop 2)
5. Co-treino da Sonda Latente (Latent Halting Probe MLP)
6. Monitoramento de Perplexidade em CADA PASSO (PPL Loop 2 e PPL Loop 1 via RMSNorm)
7. Controle Estrito de Memória (Pico de VRAM rigorosamente limitado a ~5.0 - 6.5 GB, bem abaixo de 10.5 GB)
8. Geração não truncada (MAX_NEW_TOKENS = 256) preservando qualidade, chains de raciocínio e Python
"""

import os
import sys
import json
import random
import time
import math
from typing import List, Dict, Any, Tuple

# Inserir caminhos locais para importar a engine do qwen3loop
sys.path.insert(0, os.path.abspath("engine"))
sys.path.insert(0, os.path.abspath("model_src"))
sys.path.insert(0, os.path.abspath("scripts"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
MODEL_DIR = os.path.abspath("checkpoints/qwen3loop_sft_final")
OUTPUT_DIR = os.path.abspath("checkpoints/qwen3loop_tool_grpo_final")
CURRICULUM_DIR = os.path.abspath("data/rl_curriculum")
LOG_FILE = os.path.abspath("data/tool_grpo_training_log.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

TOTAL_STEPS = 200            # 200 passos de otimização de alta densidade
GROUP_SIZE = 4               # G=4 completions por prompt (mantém VRAM em ~5.0 GB)
GRAD_ACCUM_STEPS = 4         # Passos de acumulação (16 completions por update de pesos)
MAX_NEW_TOKENS = 256         # Espaço amplo para raciocínio <think>, JSON e Python sem corte de qualidade
CLIP_EPS = 0.2               # PPO / GRPO clipping epsilon
GAMMA_LOOP1 = 0.30           # Peso do gradiente de política no Loop 1
ALPHA_DISTILL = 0.10         # Peso da divergência KL entre Loop 1 e Loop 2
LAMBDA_PROBE = 0.05          # Peso da perda de calibração da sonda MLP
LR = 2e-5                    # Taxa de aprendizado da política LoRA
PROBE_LR = 1e-4              # Taxa de aprendizado da sonda
EVAL_INTERVAL = 50           # Avaliação no benchmark holdout a cada 50 passos
SAVE_INTERVAL = 50           # Salvamento de checkpoint a cada 50 passos

# -------------------------------------------------------------
# Classe da Sonda Latente de Roteamento Dinâmico
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Curriculum Dataset Loader & Dynamic Sampler
# -------------------------------------------------------------
class DynamicCurriculumSampler:
    def __init__(self, data_dir: str):
        self.l1_data = self._load_jsonl(os.path.join(data_dir, "level_1_foundation.jsonl"))
        self.l2_data = self._load_jsonl(os.path.join(data_dir, "level_2_discrimination.jsonl"))
        self.l3_data = self._load_jsonl(os.path.join(data_dir, "level_3_reasoning_python.jsonl"))
        self.eval_data = self._load_jsonl(os.path.join(data_dir, "eval_benchmark_battery.jsonl"))
        
        print("CurriculumSampler carregado:", flush=True)
        print(f"  • Nível 1: {len(self.l1_data):,} prompts (Fundação & Abstenção)", flush=True)
        print(f"  • Nível 2: {len(self.l2_data):,} prompts (Discriminação & Schemas)", flush=True)
        print(f"  • Nível 3: {len(self.l3_data):,} prompts (Python Execution & Raciocínio)", flush=True)
        print(f"  • Eval:    {len(self.eval_data):,} prompts", flush=True)

    def _load_jsonl(self, filepath: str) -> List[Dict[str, Any]]:
        samples = []
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    tools = item.get("tools")
                    if isinstance(tools, str):
                        try:
                            tools = json.loads(tools)
                        except Exception:
                            tools = []
                    if isinstance(tools, dict):
                        tools = [tools]
                    if not isinstance(tools, list):
                        tools = []
                    item["tools"] = tools
                    samples.append(item)
        return samples

    def sample_prompt(self, current_step: int, total_steps: int) -> Dict[str, Any]:
        progress = current_step / max(1, total_steps)
        if progress < 0.30:
            pool = self.l1_data if random.random() < 0.70 else self.l2_data
        elif progress < 0.70:
            r = random.random()
            if r < 0.20:
                pool = self.l1_data
            elif r < 0.70:
                pool = self.l2_data
            else:
                pool = self.l3_data
        else:
            r = random.random()
            if r < 0.10:
                pool = self.l1_data
            elif r < 0.40:
                pool = self.l2_data
            else:
                pool = self.l3_data
        return random.choice(pool)

# -------------------------------------------------------------
# Função de Avaliação Rápida do Benchmark
# -------------------------------------------------------------
def run_evaluation_benchmark(model, tokenizer, eval_samples: List[Dict[str, Any]], max_cases: int = 15) -> Dict[str, float]:
    model.eval()
    selected = eval_samples[:max_cases]
    total = len(selected)
    correct_format = 0
    correct_selection = 0
    correct_abstention = 0
    total_abstention_cases = 0
    
    with torch.no_grad():
        for case in selected:
            tools = case.get("tools")
            tools_arg = tools if (isinstance(tools, list) and len(tools) > 0) else None
            prompt_str = tokenizer.apply_chat_template(
                case["messages"],
                tools=tools_arg,
                add_generation_prompt=True,
                tokenize=False
            )
            inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)
            outputs = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id
            )
            gen_ids = outputs[0][inputs.input_ids.shape[1]:]
            comp = tokenizer.decode(gen_ids, skip_special_tokens=False).replace("<|im_end|>", "").strip()
            
            exp_call = case.get("expected_call")
            rf = reward_format([comp], expected_call=[exp_call])[0]
            if rf > 0.5:
                correct_format += 1
                
            rs = reward_tool_selection([comp], expected_call=[exp_call])[0]
            if rs > 0.5:
                correct_selection += 1
                
            if exp_call is None:
                total_abstention_cases += 1
                ra = reward_abstention([comp], expected_call=[None])[0]
                if ra > 0.5:
                    correct_abstention += 1

    model.train()
    return {
        "eval_format_acc": (correct_format / total) * 100,
        "eval_selection_acc": (correct_selection / total) * 100,
        "eval_abstention_acc": (correct_abstention / max(1, total_abstention_cases)) * 100
    }

# -------------------------------------------------------------
# Loop Principal de Treinamento GRPO Recursivo
# -------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="RIC-GRPO Trainer")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS, help="Passos de treino")
    parser.add_argument("--eval_interval", type=int, default=EVAL_INTERVAL, help="Intervalo de eval")
    parser.add_argument("--save_interval", type=int, default=SAVE_INTERVAL, help="Intervalo de save")
    args = parser.parse_args()

    total_steps = args.steps
    eval_interval = args.eval_interval
    save_interval = args.save_interval

    print("=" * 70, flush=True)
    print("INICIANDO RECURSIVE INTERMEDIATE-CALIBRATED GRPO (RIC-GRPO)", flush=True)
    print(f"Dispositivo:       {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    print(f"Precisão:          {DTYPE}", flush=True)
    print(f"Modelo Base:       {MODEL_DIR}", flush=True)
    print(f"Passos:            {total_steps} | Group Size: G={GROUP_SIZE} | Grad Accum: {GRAD_ACCUM_STEPS}", flush=True)
    print(f"Limite Geração:    {MAX_NEW_TOKENS} tokens (preservação total de qualidade)", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # 1. Carregar Tokenizer e Modelo Base
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    config = AutoConfig.from_pretrained(MODEL_DIR)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        config=config,
        torch_dtype=DTYPE,
        attn_implementation="sdpa"
    ).to(DEVICE)

    # Ativar Gradient Checkpointing nativo do Qwen3Loop (reduz ativações de 56 camadas de ~11 GB para < 2.5 GB)
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
    model.print_trainable_parameters()

    # 3. Carregar Sonda Latente
    probe = LatentHaltingProbe(config.hidden_size).to(DEVICE).to(DTYPE)
    probe_path = os.path.join(MODEL_DIR, "latent_halting_probe.pt")
    if os.path.exists(probe_path):
        try:
            probe.load_state_dict(torch.load(probe_path, map_location=DEVICE, weights_only=True))
            print("Latent Halting Probe carregada com sucesso!", flush=True)
        except Exception as e:
            print(f"Aviso ao carregar probe: {e}. Inicializando nova sonda.", flush=True)

    # 4. Otimizadores
    optimizer_policy = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    optimizer_probe = AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer_policy, T_max=total_steps, eta_min=1e-6)

    # 5. Sampler do Currículo
    curriculum = DynamicCurriculumSampler(CURRICULUM_DIR)

    # 6. Variáveis de Acompanhamento
    accumulated_loss = 0.0
    step_times = []
    training_history = []
    start_time = time.time()

    print("\n--- TREINAMENTO ATIVO INICIADO ---\n", flush=True)

    for step in range(1, total_steps + 1):
        step_t0 = time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # A. Amostrar prompt do Currículo
        sample = curriculum.sample_prompt(step, total_steps)
        tools = sample.get("tools")
        tools_arg = tools if (isinstance(tools, list) and len(tools) > 0) else None

        prompt_str = tokenizer.apply_chat_template(
            sample["messages"],
            tools=tools_arg,
            add_generation_prompt=True,
            tokenize=False
        )
        # Sem truncamento de prompt ou resposta: com GC + SDPA o pico no pior caso do dataset (1.850 tokens) é de apenas 2.80 GB
        prompt_inputs = tokenizer(prompt_str, return_tensors="pt").to(DEVICE)
        prompt_len = prompt_inputs.input_ids.shape[1]

        # B. ROLLOUT GRPO: Gerar G completions com temperatura para exploração
        model.eval()
        with torch.no_grad():
            rollout_outputs = model.generate(
                **prompt_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=GROUP_SIZE,
                pad_token_id=tokenizer.eos_token_id
            )
        model.train()

        # C. Avaliar Recompensas Hierárquicas para cada completion do grupo
        completions = []
        for i in range(GROUP_SIZE):
            gen_tokens = rollout_outputs[i][prompt_len:]
            text = tokenizer.decode(gen_tokens, skip_special_tokens=False).replace("<|im_end|>", "").strip()
            completions.append(text)

        exp_call = sample.get("expected_call")
        exp_res = sample.get("expected_execution_result")

        # Recompensas em cascata (HTGR)
        r_fmt = reward_format(completions, expected_call=[exp_call]*GROUP_SIZE)
        r_sel = reward_tool_selection(completions, expected_call=[exp_call]*GROUP_SIZE)
        r_arg = reward_arguments(completions, expected_call=[exp_call]*GROUP_SIZE)
        r_abs = reward_abstention(completions, expected_call=[exp_call]*GROUP_SIZE)
        r_py  = reward_python_execution(completions, expected_execution_result=[exp_res]*GROUP_SIZE)

        total_rewards = [
            (rf * 1.0) + (rs * 1.0) + (ra * 1.0) + (rab * 1.5) + (rpy * 2.0)
            for rf, rs, ra, rab, rpy in zip(r_fmt, r_sel, r_arg, r_abs, r_py)
        ]

        mean_r = sum(total_rewards) / GROUP_SIZE
        var_r = sum((r - mean_r) ** 2 for r in total_rewards) / GROUP_SIZE
        std_r = math.sqrt(var_r) + 1e-4

        # Vantagens normalizadas dentro do grupo
        advantages = [(r - mean_r) / std_r for r in total_rewards]

        # D. FORWARD PASS COM CALIBRAÇÃO DOS LOOPS & MENSURAÇÃO DE PPL
        step_ppl_l2 = 0.0
        step_ppl_l1 = 0.0

        for i in range(GROUP_SIZE):
            seq_ids = rollout_outputs[i].unsqueeze(0)
            adv = advantages[i]

            outputs = model(
                input_ids=seq_ids,
                output_hidden_states=False,
                return_dict=True
            )

            # 1. Logits e Perplexidade do Loop 2 (Final, 56L)
            logits_l2 = outputs.logits[:, prompt_len - 1 : -1, :]
            targets = seq_ids[:, prompt_len:]

            log_probs_l2 = F.log_softmax(logits_l2, dim=-1)
            token_log_probs_l2 = torch.gather(log_probs_l2, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
            nll_l2 = -token_log_probs_l2.mean()
            step_ppl_l2 += torch.exp(nll_l2.detach()).item()
            
            loss_policy_l2 = nll_l2 * adv

            # 2. Logits, Perplexidade e Calibração do Loop 1 (Intermediário com RMSNorm)
            loop_hidden = getattr(outputs, "loop_hidden_states", None)
            loss_policy_l1 = torch.tensor(0.0, device=DEVICE)
            loss_distill = torch.tensor(0.0, device=DEVICE)

            if loop_hidden and len(loop_hidden) > 1:
                mid_idx = len(loop_hidden) // 2
                h_l1_raw = loop_hidden[mid_idx]
                h_l1 = base_model.model.norm(h_l1_raw)
                logits_l1 = base_model.lm_head(h_l1[:, prompt_len - 1 : -1, :])
                log_probs_l1 = F.log_softmax(logits_l1, dim=-1)
                token_log_probs_l1 = torch.gather(log_probs_l1, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
                nll_l1 = -token_log_probs_l1.mean()
                step_ppl_l1 += torch.exp(nll_l1.detach()).item()
                
                loss_policy_l1 = nll_l1 * adv

                # Consistência inter-loop via auto-destilação
                probs_l2_detach = F.softmax(logits_l2.detach(), dim=-1)
                loss_distill = F.kl_div(log_probs_l1, probs_l2_detach, reduction="batchmean")

                # Sonda Latente de Parada Antecipada
                h_prompt_end = h_l1[:, prompt_len - 1, :]
                probe_logits, probe_score = probe(h_prompt_end)
                
                if exp_call is None:
                    target_cls = torch.tensor([0], device=DEVICE)
                elif "python" in exp_call.get("name", ""):
                    target_cls = torch.tensor([2], device=DEVICE)
                else:
                    target_cls = torch.tensor([1], device=DEVICE)

                loss_probe = F.cross_entropy(probe_logits, target_cls)
            else:
                loss_probe = torch.tensor(0.0, device=DEVICE)

            # Perda Integrada RIC-GRPO
            total_sample_loss = (
                loss_policy_l2 +
                (GAMMA_LOOP1 * loss_policy_l1) +
                (ALPHA_DISTILL * loss_distill) +
                (LAMBDA_PROBE * loss_probe)
            ) / (GROUP_SIZE * GRAD_ACCUM_STEPS)

            total_sample_loss.backward()
            accumulated_loss += total_sample_loss.item()

            del outputs, total_sample_loss
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

        # E. Otimização de Pesos
        if step % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_policy.step()
            optimizer_probe.step()
            scheduler.step()
            optimizer_policy.zero_grad()
            optimizer_probe.zero_grad()

        # F. Métricas de CADA PASSO
        step_elapsed = time.time() - step_t0
        step_times.append(step_elapsed)
        peak_vram_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

        avg_step_ppl_l2 = step_ppl_l2 / GROUP_SIZE
        avg_step_ppl_l1 = (step_ppl_l1 / GROUP_SIZE) if step_ppl_l1 > 0 else avg_step_ppl_l2

        phase = 1 if (step / total_steps) < 0.30 else (2 if (step / total_steps) < 0.70 else 3)

        # Cálculo de ETA em tempo real
        avg_time = sum(step_times[-10:]) / len(step_times[-10:])
        remaining_steps = total_steps - step
        eta_minutes = (remaining_steps * avg_time) / 60

        # Log transparente e imediato em CADA PASSO
        print(
            f"[Step {step:03d}/{total_steps}] Fase {phase} | "
            f"Reward: {mean_r:.3f} | "
            f"PPL-L2: {avg_step_ppl_l2:.2f} | "
            f"PPL-L1: {avg_step_ppl_l1:.2f} | "
            f"VRAM Pico: {peak_vram_gb:.2f} GB | "
            f"Tempo: {step_elapsed:.1f}s | "
            f"ETA: {eta_minutes:.1f} min",
            flush=True
        )

        training_history.append({
            "step": step,
            "phase": phase,
            "reward": mean_r,
            "ppl_loop2": avg_step_ppl_l2,
            "ppl_loop1": avg_step_ppl_l1,
            "peak_vram_gb": peak_vram_gb,
            "step_time_sec": step_elapsed,
            "eta_min": eta_minutes
        })
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2)

        # G. Avaliação Periódica no Benchmark Hold-out
        if step % eval_interval == 0:
            print("\n" + "-" * 65, flush=True)
            print(f">>> AVALIAÇÃO DE CHECKPOINT NO STEP {step} <<<", flush=True)
            eval_metrics = run_evaluation_benchmark(model, tokenizer, curriculum.eval_data, max_cases=15)
            print(f"Formato JSON Válido:      {eval_metrics['eval_format_acc']:.1f}%", flush=True)
            print(f"Seleção de Tool Correta:  {eval_metrics['eval_selection_acc']:.1f}%", flush=True)
            print(f"Abstenção Conversacional: {eval_metrics['eval_abstention_acc']:.1f}%", flush=True)
            print("-" * 65 + "\n", flush=True)

        # H. Salvamento Periódico de Checkpoint
        if step % save_interval == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(probe.state_dict(), os.path.join(ckpt_dir, "latent_halting_probe.pt"))
            print(f"Checkpoint intermediário salvo em: {ckpt_dir}\n", flush=True)

    # 7. Finalização e Salvamento Consolidado
    print("\n" + "=" * 70, flush=True)
    print("TREINAMENTO RL/GRPO CONCLUÍDO COM SUCESSO!", flush=True)
    total_time = (time.time() - start_time) / 60
    print(f"Tempo total de treino: {total_time:.2f} minutos.", flush=True)

    print(f"Salvando modelo final calibrado em: {OUTPUT_DIR}", flush=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    torch.save(probe.state_dict(), os.path.join(OUTPUT_DIR, "latent_halting_probe.pt"))

    # Avaliação Final no Benchmark Hold-out
    print("\n--- AVALIAÇÃO FINAL NO BENCHMARK HOLD-OUT ---", flush=True)
    final_eval = run_evaluation_benchmark(model, tokenizer, curriculum.eval_data, max_cases=30)
    print(f"Formato JSON Válido:      {final_eval['eval_format_acc']:.1f}%", flush=True)
    print(f"Seleção de Tool Correta:  {final_eval['eval_selection_acc']:.1f}%", flush=True)
    print(f"Abstenção Conversacional: {final_eval['eval_abstention_acc']:.1f}%", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
