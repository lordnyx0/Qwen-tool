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
from scaffold_pool import (
    build_refine_prompt,
    ensure_files as scaffold_ensure_files,
    update as scaffold_update,
)
from rewards_tool_calling import (
    reward_format,
    reward_tool_selection,
    reward_arguments,
    reward_abstention,
    reward_python_execution,
    reward_length,
    reward_omission,
    detect_violations,
    masked_group_stats,
    apply_exclusion,
    judge_veto,
    make_llama_judge,
    make_plumb_judge,
    append_verdict_log,
    load_shell_golden,
    reward_shell_execution,
    reward_command_fidelity,
    VIOLATION_REWARD,
)

# -------------------------------------------------------------
# Configurações Globais e Hiperparâmetros
# -------------------------------------------------------------
MODEL_DIR = os.path.abspath("checkpoints/qwen3loop_sft_final")
OUTPUT_DIR = os.path.abspath("checkpoints/qwen3loop_tool_grpo_v3")
CURRICULUM_DIR = os.path.abspath("data/rl_curriculum")
LOG_FILE = os.path.abspath("data/tool_grpo_v3_training_log.json")
PROBE_FILE = os.path.abspath("data/eval_probe_canonical_25.jsonl")
BETA_KL = 0.10                # Ancora KL(policy||ref) reforçada (v4 derivou com 0.05)
LEN_W = 0.10                  # Peso da penalidade de verbosidade

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

TOTAL_STEPS = 150            # 150 passos de otimização de alta densidade
GROUP_SIZE = 4               # G=4 completions por prompt (mantém VRAM em ~5.0 GB)
GRAD_ACCUM_STEPS = 4         # Passos de acumulação (16 completions por update de pesos)
MAX_NEW_TOKENS = 256         # Espaço amplo para raciocínio <think>, JSON e Python sem corte de qualidade
SCAF_TOKENS = 64             # Orcamento do estagio A (scaffold) quando --scaffold
CLIP_EPS = 0.2               # PPO / GRPO clipping epsilon
GAMMA_LOOP1 = 0.0            # Desligado: mid-layer nao deve predizer direto; so distill L1->L2
MID_KL = 0.10                # Ancora KL(mid_policy||mid_ref congelada): sem ela os mids derivam livres (ppl_loop1 em spikes) e arrastam o tronco
ALPHA_DISTILL = 0.10         # Peso da divergência KL entre Loop 1 e Loop 2
LAMBDA_PROBE = 0.05          # Peso da perda de calibração da sonda MLP
LR = 2e-5                    # Taxa de aprendizado da política LoRA
PROBE_LR = 1e-4              # Taxa de aprendizado da sonda
EVAL_INTERVAL = 25           # Avaliação no benchmark holdout a cada 25 passos (v5-safe)
SAVE_INTERVAL = 25           # Pool p/ selecao post-hoc do melhor (nao so o final)

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


def _migrate_probe_state(probe, probe_path):
    """Migra sonda legada net.{0,3} (Linear+GELU+Dropout+Linear) p/ cabeca atual.

    Unico mapeamento exato de shapes: net.3 (64->3, sem bias) -> classifier.
    Retorna lista das chaves migradas ([] se nada compativel). Puro em I/O,
    sem modelo: testavel em CPU.
    """
    try:
        old = torch.load(probe_path, map_location="cpu", weights_only=True)
    except Exception:
        return []
    migrated = []
    try:
        new_sd = probe.state_dict()
    except Exception:
        return []
    if ("net.3.weight" in old and "classifier.weight" in new_sd
            and tuple(old["net.3.weight"].shape) == tuple(new_sd["classifier.weight"].shape)):
        with torch.no_grad():
            new_sd["classifier.weight"].copy_(old["net.3.weight"].to(new_sd["classifier.weight"].dtype))
        probe.load_state_dict(new_sd, strict=False)
        migrated.append("net.3.weight->classifier.weight")
    return migrated

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
# FrontierSampler V3: pools por level do rollout (pesos da tabela V3)
# abstain .30 / python .30 / discrimination .25 / multiturn .15
# -------------------------------------------------------------
class FrontierSampler:
    WEIGHTS = [("frontier_abstain", 0.30), ("frontier_python", 0.30),
               ("frontier_discrimination", 0.25), ("frontier_multiturn", 0.15)]
    GENERIC_WEIGHTS = [("ws_powershell", 0.45), ("abstain", 0.20),
                       ("discrim_shell", 0.20), ("shell_multi", 0.15)]

    def __init__(self, rollout_path: str, probe_path: str):
        self.pools = {}
        self.probe_data = []
        for line in open(rollout_path, encoding="utf-8"):
            if line.strip():
                it = json.loads(line)
                self.pools.setdefault(it.get("level", "?"), []).append(it)
        try:
            with open(probe_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.probe_data.append(json.loads(line))
        except Exception as e:
            print(f"AVISO probe indisponivel ({probe_path}): {e}; "
                  f"seguindo sem monitoramento.", flush=True)
        total = sum(len(v) for v in self.pools.values())
        print(f"Frontier pools: { {k: len(v) for k, v in self.pools.items()} } "
              f"total={total} | probe={len(self.probe_data)}", flush=True)
        assert total > 0, "rollout vazio!"
        keys = set(self.pools)
        if keys and not any(k.startswith("frontier_") for k in keys):
            self.weights = list(self.GENERIC_WEIGHTS)
        else:
            self.weights = list(self.WEIGHTS)
        print(f"Sampler weights: {self.weights}", flush=True)

    def sample_prompt(self, current_step: int, total_steps: int) -> Dict[str, Any]:
        r = random.random()
        acc = 0.0
        for lvl, w in self.weights:
            acc += w
            if r < acc and self.pools.get(lvl):
                return random.choice(self.pools[lvl])
        pool = random.choice([v for v in self.pools.values() if v])
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
            if exp_call is None and case.get("expected_tool"):
                exp_args = case.get("expected_args") or {}
                if isinstance(exp_args, str):
                    try:
                        exp_args = json.loads(exp_args)
                    except Exception:
                        exp_args = {}
                exp_call = {"name": case.get("expected_tool"), "arguments": exp_args}
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
    global MODEL_DIR, OUTPUT_DIR, GROUP_SIZE
    import argparse
    parser = argparse.ArgumentParser(description="RIC-GRPO Trainer")
    parser.add_argument("--steps", type=int, default=TOTAL_STEPS, help="Passos de treino")
    parser.add_argument("--group_size", type=int, default=GROUP_SIZE,
                        help="G completions por prompt (VRAM ~plana em G; tempo ~linear)")
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--rollouts", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--beta_kl", type=float, default=BETA_KL)
    parser.add_argument("--mid_kl", type=float, default=MID_KL)
    parser.add_argument("--len_w", type=float, default=LEN_W)
    parser.add_argument("--probe", type=str, default=None)
    parser.add_argument("--eval_interval", type=int, default=EVAL_INTERVAL, help="Intervalo de eval")
    parser.add_argument("--save_interval", type=int, default=SAVE_INTERVAL, help="Intervalo de save")
    parser.add_argument("--judge_veto", action="store_true",
                        help="Liga veto do juiz congelado em casos suspeitos")
    parser.add_argument("--judge_backend", choices=["llama", "plumb"], default="llama")
    parser.add_argument("--judge_url", type=str, default=None)
    parser.add_argument("--judge_threshold", type=float, default=0.8,
                        help="P(true) de gaming p/ veto (so backend plumb)")
    parser.add_argument("--scaffold", action="store_true",
                        help="Liga rollout two-stage Ornith (A: scaffold, B: solucao)")
    parser.add_argument("--scaffold_tokens", type=int, default=SCAF_TOKENS)
    parser.add_argument("--shell_golden", type=str, default=None,
                        help="Tabela golden {(tool,command): stdout} p/ reward_shell (off sem path)")
    parser.add_argument("--judge_log", type=str, default=None,
                        help="JSONL p/ destilar o veto depois (etapa Kaggle; off sem path)")
    parser.add_argument("--rollout_log", type=str, default=None,
                        help="JSONL por step: scaffolds, completions, rewards, vantagens (calibracao)")
    args = parser.parse_args()

    if args.model_dir:
        MODEL_DIR = os.path.abspath(args.model_dir)
    GROUP_SIZE = max(2, args.group_size)
    if args.output_dir:
        OUTPUT_DIR = os.path.abspath(args.output_dir)
    rollouts_path = (os.path.abspath(args.rollouts) if args.rollouts
                     else os.path.abspath("data/rl_rollouts/grpo_frontier_v1.jsonl"))
    probe_json_path = (os.path.abspath(args.probe) if args.probe else PROBE_FILE)
    beta_kl = args.beta_kl
    mid_kl_w = args.mid_kl
    len_w = args.len_w
    assert mid_kl_w > 0, "mid_kl zerado: mids sem ancora (deriva livre)"

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
    with torch.no_grad():
        ref_model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, config=config, torch_dtype=DTYPE,
            attn_implementation="sdpa").to(DEVICE)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    print("Modelo de referencia congelado p/ KL.", flush=True)
    model.print_trainable_parameters()

    # 3. Carregar Sonda Latente (direto -> migracao parcial net.* -> classifier -> nova)
    probe = LatentHaltingProbe(config.hidden_size).to(DEVICE).to(DTYPE)
    probe_path = os.path.join(MODEL_DIR, "latent_halting_probe.pt")
    if os.path.exists(probe_path):
        try:
            probe.load_state_dict(torch.load(probe_path, map_location=DEVICE, weights_only=True))
            print("Latent Halting Probe carregada com sucesso!", flush=True)
        except Exception as e:
            migrated = _migrate_probe_state(probe, probe_path)
            if migrated:
                print(f"Probe migrada parcial ({migrated}): {e}.", flush=True)
            else:
                print(f"Aviso ao carregar probe: {e}. Inicializando nova sonda.", flush=True)

    # 4. Otimizadores
    optimizer_policy = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    optimizer_probe = AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer_policy, T_max=total_steps, eta_min=1e-6)

    # 5. Sampler do Currículo
    curriculum = FrontierSampler(rollouts_path, probe_json_path)
    print(f"Modelo: {MODEL_DIR} | beta_kl={beta_kl} len_w={len_w}", flush=True)
    use_scaffold = args.scaffold
    scaf_tokens = args.scaffold_tokens
    if use_scaffold:
        scaffold_ensure_files()
        print(f"Scaffold two-stage LIGADO (A={scaf_tokens}tk). Pool em data/rl_scaffolds/.", flush=True)

    # 6. Variáveis de Acompanhamento
    accumulated_loss = 0.0
    step_times = []
    training_history = []
    start_time = time.time()

    print("\n--- TREINAMENTO ATIVO INICIADO ---\n", flush=True)

    # Estado dos guard-rails v5-safe (calibração)
    ppl_bad_streak = 0
    excl_bad_streak = 0
    judge_cache = {}
    judge_fn = None
    if args.judge_veto:
        if args.judge_backend == "plumb":
            judge_fn = make_plumb_judge(base_url=args.judge_url or "http://127.0.0.1:8090",
                                        threshold=args.judge_threshold)
        else:
            judge_fn = make_llama_judge(base_url=args.judge_url or "http://127.0.0.1:8080")
        print(f"Juiz veto LIGADO backend={args.judge_backend} url={args.judge_url or 'default'}.", flush=True)
    judge_log_fp = None
    if args.judge_veto and args.judge_log:
        judge_log_fp = open(args.judge_log, "w", encoding="utf-8")
    rollout_log_fp = None
    if args.rollout_log:
        rollout_log_fp = open(args.rollout_log, "w", encoding="utf-8")
    shell_golden = load_shell_golden(args.shell_golden)
    if args.shell_golden:
        print(f"Shell golden: {len(shell_golden)} entradas.", flush=True)
    nocall_hist = []
    nocall_base = 0.0
    best_probe_overall = -1.0
    best_probe_step = 0

    def _save_ckpt(tag):
        d = os.path.join(OUTPUT_DIR, f"checkpoint-{tag}")
        os.makedirs(d, exist_ok=True)
        model.save_pretrained(d)
        tokenizer.save_pretrained(d)
        torch.save(probe.state_dict(), os.path.join(d, "latent_halting_probe.pt"))
        return d

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
        # Two-stage Ornith (--scaffold): A gera scaffold refinado; B resolve
        # condicionado nele. Um reward unico flui aos tokens de A+B (joint).
        user_text = " ".join(m.get("content", "") for m in sample.get("messages", [])
                             if m.get("role") == "user")
        level = sample.get("level", "?")
        model.eval()
        all_seqs = []
        all_splits = []
        scaffold_texts = [""] * GROUP_SIZE
        with torch.no_grad():
            if use_scaffold:
                scaf_prompt = build_refine_prompt(user_text, level)
                scaf_inputs = tokenizer(scaf_prompt, return_tensors="pt").to(DEVICE)
                scaf_out = model.generate(
                    **scaf_inputs,
                    max_new_tokens=scaf_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=GROUP_SIZE,
                    pad_token_id=tokenizer.eos_token_id
                )
                scaf_len = scaf_inputs.input_ids.shape[1]
                for i in range(GROUP_SIZE):
                    st = tokenizer.decode(scaf_out[i][scaf_len:], skip_special_tokens=True)
                    st = st.replace("<tool_call>", "").replace("</tool_call>", "").strip()[:600]
                    scaffold_texts[i] = st
                # B em batch unico (left-pad): 1 generate p/ G solucoes.
                sol_texts = [prompt_str + "\n" + st + "\n" for st in scaffold_texts]
                sol_ids = [tokenizer(t, return_tensors="pt").input_ids[0] for t in sol_texts]
                sol_lens = [len(s) for s in sol_ids]
                pad_id = tokenizer.eos_token_id
                max_l = max(sol_lens)
                batch_ids = torch.stack([
                    torch.cat([torch.full((max_l - len(s),), pad_id, dtype=torch.long), s])
                    for s in sol_ids]).to(DEVICE)
                batch_mask = (batch_ids != pad_id).long()
                sol_batch = model.generate(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=pad_id
                )
                for i in range(GROUP_SIZE):
                    gen = sol_batch[i][max_l:]
                    full = torch.cat([sol_ids[i].to(DEVICE), gen])
                    all_seqs.append(full)
                    all_splits.append(sol_lens[i])
            else:
                rollout_outputs = model.generate(
                    **prompt_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=GROUP_SIZE,
                    pad_token_id=tokenizer.eos_token_id
                )
                for i in range(GROUP_SIZE):
                    all_seqs.append(rollout_outputs[i])
                    all_splits.append(prompt_len)
        model.train()

        # C. Avaliar Recompensas Hierárquicas para cada completion do grupo
        completions = []
        for i in range(GROUP_SIZE):
            gen_tokens = all_seqs[i][all_splits[i]:]
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
        r_len = reward_length(completions, max_tokens=MAX_NEW_TOKENS)
        r_omi = reward_omission(completions, expected_call=[exp_call]*GROUP_SIZE)
        r_sh = reward_shell_execution(completions, samples=[sample]*GROUP_SIZE,
                                      golden=shell_golden,
                                      expected_call=[exp_call]*GROUP_SIZE)
        r_fid = reward_command_fidelity(completions, expected_call=[exp_call]*GROUP_SIZE)

        total_rewards = [
            (rf * 1.0) + (rs * 1.0) + (ra * 1.0) + (rab * 1.5) + (rpy * 2.0) + (rl * len_w) + (ro * 1.0) + (rsh * 2.0) + (rfid * 1.0)
            for rf, rs, ra, rab, rpy, rl, ro, rsh, rfid in zip(r_fmt, r_sel, r_arg, r_abs, r_py, r_len, r_omi, r_sh, r_fid)
        ]
        # Monitor Ornith-L1: violacao -> 0.0 + exclusao do computo de vantagem.
        declared_set = {t.get("function", {}).get("name") for t in (tools or [])
                        if isinstance(t, dict)} if tools_arg else set()
        excluded = detect_violations(completions, declared=[declared_set] * GROUP_SIZE,
                                     expected_call=[exp_call] * GROUP_SIZE)
        total_rewards = [VIOLATION_REWARD if x else r
                         for r, x in zip(total_rewards, excluded)]
        n_excluded = sum(1 for x in excluded if x)
        # Juiz Ornith-L2 (opt-in): veto em suspeitos nao-excluidos, com cache.
        n_vetoed = 0
        if judge_fn is not None:
            user_text = " ".join(m.get("content", "") for m in sample.get("messages", [])
                                 if m.get("role") == "user")
            todo_idx = [i for i, x in enumerate(excluded) if not x]

            def _cached(prompt_text):
                if prompt_text not in judge_cache:
                    judge_cache[prompt_text] = judge_fn(prompt_text)
                    append_verdict_log(judge_log_fp, step, prompt_text,
                                       judge_cache[prompt_text])
                return judge_cache[prompt_text]

            veto_flags = judge_veto(
                [completions[i] for i in todo_idx],
                expected_call=[exp_call] * len(todo_idx),
                tasks=[user_text] * len(todo_idx),
                judge_fn=_cached)
            for i, v in zip(todo_idx, veto_flags):
                if v:
                    excluded[i] = True
                    total_rewards[i] = VIOLATION_REWARD
                    n_vetoed += 1
            n_excluded = sum(1 for x in excluded if x)
        # Pool Ornith: persiste vencedores por nivel (so com --scaffold;
        # excluidas nao ensinam plano).
        if use_scaffold:
            for i in range(GROUP_SIZE):
                if not excluded[i] and scaffold_texts[i]:
                    is_best, best_mean = scaffold_update(level, scaffold_texts[i],
                                                        total_rewards[i])
                    if is_best:
                        print(f"[scaffold] novo best @{level} mean={best_mean:.3f}", flush=True)
        nocall_frac = sum(1.0 for x in r_omi if x < 0.0) / GROUP_SIZE
        is_py_item = exp_res is not None
        hits = sum(
            1 for _rf, _rs, _ra, _rpy in zip(r_fmt, r_sel, r_arg, r_py)
            if _rf >= 1.0 and _rs >= 1.0 and _ra >= 1.0
            and (not is_py_item or _rpy >= 2.0)
        )

        mean_r, std_r, all_excluded = masked_group_stats(total_rewards, excluded)

        # Vantagens normalizadas dentro do grupo (excluidas zeradas: sem gradiente)
        advantages = apply_exclusion([(r - mean_r) / std_r for r in total_rewards], excluded)

        # Log total p/ analise de calibracao (tudo que o modelo gerou + resultados).
        if rollout_log_fp is not None:
            rollout_log_fp.write(json.dumps({
                "step": step, "level": sample.get("level", "?"),
                "sample_id": sample.get("id", sample.get("uuid", "?")),
                "scaffolds": scaffold_texts,
                "completions": completions,
                "rewards": {"format": r_fmt, "selection": r_sel, "arguments": r_arg,
                            "abstention": r_abs, "python": r_py, "length": r_len,
                            "omission": r_omi, "fidelity": r_fid,
                            "shell": r_sh if "r_sh" in dir() else []},
                "total": total_rewards, "advantages": advantages,
                "excluded": excluded,
                "vetoed": [bool(v) for v in veto_flags] if "veto_flags" in dir() else [],
            }, ensure_ascii=False) + "\n")
            rollout_log_fp.flush()

        # D. FORWARD PASS COM CALIBRAÇÃO DOS LOOPS & MENSURAÇÃO DE PPL
        step_ppl_l2 = 0.0
        step_ppl_l1 = 0.0

        for i in range(GROUP_SIZE):
            seq_ids = all_seqs[i].unsqueeze(0)
            split = all_splits[i]
            adv = advantages[i]

            outputs = model(
                input_ids=seq_ids,
                output_hidden_states=False,
                return_dict=True
            )

            # 1. Logits e Perplexidade do Loop 2 (Final, 56L)
            logits_l2 = outputs.logits[:, split - 1 : -1, :]
            targets = seq_ids[:, split:]

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
                logits_l1 = base_model.lm_head(h_l1[:, split - 1 : -1, :])
                log_probs_l1 = F.log_softmax(logits_l1, dim=-1)
                token_log_probs_l1 = torch.gather(log_probs_l1, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
                nll_l1 = -token_log_probs_l1.mean()
                step_ppl_l1 += torch.exp(nll_l1.detach()).item()
                
                loss_policy_l1 = nll_l1 * adv

                # Consistência inter-loop via auto-destilação
                probs_l2_detach = F.softmax(logits_l2.detach(), dim=-1)
                loss_distill = F.kl_div(log_probs_l1, probs_l2_detach, reduction="batchmean")

                # Sonda Latente de Parada Antecipada
                h_prompt_end = h_l1[:, split - 1, :]
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
            with torch.no_grad():
                ref_out = ref_model(input_ids=seq_ids, return_dict=True)
                ref_lp = F.log_softmax(
                    ref_out.logits[:, split - 1: -1, :], dim=-1)
            kl_pr = (torch.exp(token_log_probs_l2) *
                     (token_log_probs_l2 - torch.gather(
                         ref_lp, dim=-1,
                         index=targets.unsqueeze(-1)).squeeze(-1))).mean()
            # Ancora dos mids na ref congelada (mesmo mid_idx da policy; sem grad na ref)
            kl_mid = torch.tensor(0.0, device=DEVICE)
            ref_loop = getattr(ref_out, "loop_hidden_states", None)
            if loop_hidden and not ref_loop:
                raise RuntimeError("ref sem loop_hidden_states: KL dos mids impossivel")
            if loop_hidden and ref_loop and len(loop_hidden) > 1 and len(ref_loop) > mid_idx:
                with torch.no_grad():
                    h_ref_raw = ref_loop[mid_idx]
                    h_ref = base_model.model.norm(h_ref_raw)
                    ref_mid_lp = F.log_softmax(
                        base_model.lm_head(h_ref[:, split - 1: -1, :]), dim=-1)
                kl_mid = (torch.exp(token_log_probs_l1.detach()) *
                          (token_log_probs_l1 - torch.gather(
                              ref_mid_lp, dim=-1,
                              index=targets.unsqueeze(-1)).squeeze(-1))).mean()
            total_sample_loss = (
                loss_policy_l2 +
                (GAMMA_LOOP1 * loss_policy_l1) +
                (ALPHA_DISTILL * loss_distill) +
                (LAMBDA_PROBE * loss_probe) +
                (beta_kl * kl_pr) +
                (mid_kl_w * kl_mid)
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
            f"Acertos: {hits}/{GROUP_SIZE} | "
            f"Excl: {n_excluded}/{GROUP_SIZE} | "
            f"Veto: {n_vetoed}/{GROUP_SIZE} | "
            f"PPL-L2: {avg_step_ppl_l2:.2f} | "
            f"PPL-L1: {avg_step_ppl_l1:.2f} | "
            f"NoCall: {nocall_frac:.2f} | "
            f"VRAM Pico: {peak_vram_gb:.2f} GB | "
            f"Tempo: {step_elapsed:.1f}s | "
            f"ETA: {eta_minutes:.1f} min",
            flush=True
        )

        training_history.append({
            "step": step,
            "phase": phase,
            "reward": mean_r,
            "hits": hits,
            "excluded": n_excluded,
            "vetoed": n_vetoed,
            "ppl_loop2": avg_step_ppl_l2,
            "ppl_loop1": avg_step_ppl_l1,
            "nocall_frac": nocall_frac,
            "peak_vram_gb": peak_vram_gb,
            "step_time_sec": step_elapsed,
            "eta_min": eta_minutes
        })
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=2)

        # F2. Guard-rail PPL (v5-safe): verborragia da v4 chegou a PPL-L2 2852
        if avg_step_ppl_l2 > 50.0:
            ppl_bad_streak += 1
        else:
            ppl_bad_streak = 0
        if ppl_bad_streak >= 10:
            print(f"ABORT: PPL-L2 > 50 por 10 passos seguidos (atual {avg_step_ppl_l2:.1f}).", flush=True)
            _save_ckpt(step)
            print("Checkpoint de aborto salvo. Encerrando.", flush=True)
            break

        # F2b. Guard-rail monitor Ornith-L1: grupo todo excluido = colapso/hacking.
        if all_excluded:
            excl_bad_streak += 1
        else:
            excl_bad_streak = 0
        if excl_bad_streak >= 5:
            print(f"ABORT: 5 steps seguidos com grupo 100% excluido pelo monitor.", flush=True)
            _save_ckpt(step)
            print("Checkpoint de aborto salvo. Encerrando.", flush=True)
            break

        # F3. Guard-rail omissão (v5-safe): over-abstenção da v4 (multiturn 4x `gerou []`)
        nocall_hist.append(nocall_frac)
        if step == 10:
            nocall_base = sum(nocall_hist) / max(1, len(nocall_hist))
        if step > 15 and nocall_frac > max(2.0 * nocall_base, 0.6):
            print(f"ABORT: no-call {nocall_frac:.2f} >> baseline {nocall_base:.2f}.", flush=True)
            _save_ckpt(step)
            print("Checkpoint de aborto salvo. Encerrando.", flush=True)
            break

        # G. Avaliação Periódica no Benchmark Hold-out
        if step % eval_interval == 0 and curriculum.probe_data:
            print("\n" + "-" * 65, flush=True)
            print(f">>> AVALIAÇÃO DE CHECKPOINT NO STEP {step} <<<", flush=True)
            eval_metrics = run_evaluation_benchmark(model, tokenizer, curriculum.probe_data, max_cases=15)
            print(f"Formato JSON Válido:      {eval_metrics['eval_format_acc']:.1f}%", flush=True)
            print(f"Seleção de Tool Correta:  {eval_metrics['eval_selection_acc']:.1f}%", flush=True)
            print(f"Abstenção Conversacional: {eval_metrics['eval_abstention_acc']:.1f}%", flush=True)
            print("-" * 65 + "\n", flush=True)

            # G2. Guard-rails de sonda (v5-safe): platô e regressão
            eval_overall = (eval_metrics['eval_format_acc']
                            + eval_metrics['eval_selection_acc']
                            + eval_metrics['eval_abstention_acc']) / 3.0
            if step > 50 and eval_overall < best_probe_overall - 10.0:
                print(f"ABORT: sonda caiu >10pts do melhor "
                      f"({eval_overall:.1f}% vs {best_probe_overall:.1f}%).", flush=True)
                _save_ckpt(step)
                print("Checkpoint de aborto salvo. Encerrando.", flush=True)
                break
            if eval_overall > best_probe_overall:
                best_probe_overall = eval_overall
                best_probe_step = step
            print(f"Probe overall: {eval_overall:.1f}% "
                  f"(best {best_probe_overall:.1f}% @ step {best_probe_step})", flush=True)
            if step >= 75 and step - best_probe_step >= 50:
                print("EARLY-STOP: 50 passos sem melhora de sonda. Encerrando.", flush=True)
                break

        # H. Salvamento Periódico de Checkpoint
        if step % save_interval == 0:
            ckpt_dir = os.path.join(OUTPUT_DIR, f"checkpoint-{step}")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            torch.save(probe.state_dict(), os.path.join(ckpt_dir, "latent_halting_probe.pt"))
            print(f"Checkpoint intermediário salvo em: {ckpt_dir}\n", flush=True)

    # 7. Finalização e Salvamento Consolidado
    if judge_log_fp is not None:
        judge_log_fp.close()
        print(f"Log de vereditos do juiz fechado ({len(judge_cache)} consultas unicas).", flush=True)
    if rollout_log_fp is not None:
        rollout_log_fp.close()
        print("Rollout log fechado.", flush=True)
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
    if curriculum.probe_data:
        final_eval = run_evaluation_benchmark(model, tokenizer, curriculum.probe_data, max_cases=25)
        print(f"Formato JSON Válido:      {final_eval['eval_format_acc']:.1f}%", flush=True)
        print(f"Seleção de Tool Correta:  {final_eval['eval_selection_acc']:.1f}%", flush=True)
        print(f"Abstenção Conversacional: {final_eval['eval_abstention_acc']:.1f}%", flush=True)
    else:
        print("(sem dados de sonda; avaliação final pulada)", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
