"""
analyze_loop_contribution.py
Analisa a distribuição de contribuição dos loops (Loop 1 vs Loop 2)
no último checkpoint treinado do SFT (checkpoints/qwen3loop_tool_sft_final).

Métricas extraídas:
1. Similaridade de Cosseno entre ativações h^(1) e h^(2)
2. Magnitude da perturbação residual ||h^(2) - h^(1)|| / ||h^(1)||
3. Concordância de predição de tokens (Agreement Rate) e Taxa de Flip (Refinamento)
4. Redução de Entropia (Ganho de Confiança do Loop 2)
5. Distribuição de Decisões da Sonda Latente (Latent Halting Probe: Early-Exit vs 2 Loops)
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", "qwen3loop_tool_sft_final")
BATTERY_PATH = os.path.join(ROOT_DIR, "data", "eval_benchmark_battery_v2.jsonl")

# Carregar engine qwen3loop_python
engine_dir = os.path.join(ROOT_DIR, "engine")
if engine_dir not in sys.path:
    sys.path.insert(0, engine_dir)

try:
    import qwen3loop_python
    from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM
    AutoConfig.register("qwen3loop", Qwen3LoopConfig)
    AutoModelForCausalLM.register(Qwen3LoopConfig, Qwen3LoopForCausalLM)
    print("Engine qwen3loop_python carregada com sucesso!")
except Exception as e:
    print(f"Erro ao registrar engine qwen3loop: {e}")
    sys.exit(1)

# Definição da Sonda Latente
class LatentHaltingProbe(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 64, bias=True),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes, bias=False)
        )
    def forward(self, x):
        return self.net(x)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    print("=" * 75)
    print("DIAGNÓSTICO MECANÍSTICO DE CONTRIBUIÇÃO DE LOOPS (QWEN3LOOP SFT FINAL)")
    print(f"Dispositivo: {device} ({dtype})")
    print(f"Diretório do Checkpoint: {CHECKPOINT_DIR}")
    print("=" * 75)

    print("Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)

    print("Carregando Modelo Qwen3Loop com pesos treinados...")
    config = AutoConfig.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR,
        config=config,
        torch_dtype=dtype
    ).to(device)
    model.eval()

    # Carregar Sonda Latente
    probe_path = os.path.join(CHECKPOINT_DIR, "latent_halting_probe.pt")
    probe_head = None
    if os.path.exists(probe_path):
        print("Carregando Latent Halting Probe calibrado...")
        probe_head = LatentHaltingProbe(hidden_size=config.hidden_size, num_classes=3).to(device).to(dtype)
        probe_head.load_state_dict(torch.load(probe_path, map_location=device))
        probe_head.eval()
    else:
        print("Aviso: latent_halting_probe.pt não encontrado.")

    # Carregar amostras representativas da Bateria v2
    with open(BATTERY_PATH, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    # Selecionar até 10 amostras por categoria para análise estatística densa
    cases_by_cat = defaultdict(list)
    for c in cases:
        cases_by_cat[c.get("category", "unknown")].append(c)

    sample_cases = []
    for cat, items in cases_by_cat.items():
        sample_cases.extend(items[:8])

    print(f"\nAmostras selecionadas para diagnóstico: {len(sample_cases)} itens em {len(cases_by_cat)} categorias.")
    print("-" * 75)

    category_metrics = defaultdict(lambda: {
        "cosine_sim": [],
        "residual_norm_ratio": [],
        "token_agreement_pct": [],
        "entropy_reduction": [],
        "probe_decisions": defaultdict(int),
        "total_tokens": 0
    })

    with torch.no_grad():
        for idx, case in enumerate(sample_cases, 1):
            cat = case.get("category", "unknown")
            tools = case.get("tools", [])
            messages = case.get("messages", [])

            # Renderizar prompt
            if tools:
                prompt_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            enc = tokenizer(prompt_text, return_tensors="pt")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

            loop_hidden = getattr(outputs, "loop_hidden_states", None)
            if not loop_hidden or len(loop_hidden) < 2:
                continue

            h1 = loop_hidden[0]  # [batch, seq_len, hidden_size]
            h2 = loop_hidden[1]  # [batch, seq_len, hidden_size]

            # 1. Similaridade de Cosseno entre h1 e h2
            cos_sim = F.cosine_similarity(h1, h2, dim=-1).mean().item()

            # 2. Ratio da Norma Residual: ||h2 - h1|| / ||h1||
            diff_norm = torch.norm(h2 - h1, dim=-1).mean()
            h1_norm = torch.norm(h1, dim=-1).mean()
            norm_ratio = (diff_norm / (h1_norm + 1e-8)).item()

            # 3. Logits no Head do Modelo
            logits1 = model.lm_head(h1)  # [batch, seq_len, vocab_size]
            logits2 = model.lm_head(h2)  # [batch, seq_len, vocab_size]

            top1_1 = logits1.argmax(dim=-1)
            top1_2 = logits2.argmax(dim=-1)
            agreement = (top1_1 == top1_2).float().mean().item() * 100

            # 4. Entropia de Shannon (Confiança / Incerteza)
            p1 = F.softmax(logits1.float(), dim=-1)
            p2 = F.softmax(logits2.float(), dim=-1)
            ent1 = -(p1 * torch.log(p1 + 1e-12)).sum(dim=-1).mean().item()
            ent2 = -(p2 * torch.log(p2 + 1e-12)).sum(dim=-1).mean().item()
            entropy_reduction = ent1 - ent2  # Positivo = Loop 2 reduziu incerteza (mais confiante)

            # 5. Sonda Latente no último token de prompt
            probe_pred = -1
            if probe_head is not None:
                last_token_h1 = h1[:, -1, :].to(dtype)
                probe_logits = probe_head(last_token_h1)
                probe_pred = probe_logits.argmax(dim=-1).item()
                category_metrics[cat]["probe_decisions"][probe_pred] += 1

            category_metrics[cat]["cosine_sim"].append(cos_sim)
            category_metrics[cat]["residual_norm_ratio"].append(norm_ratio)
            category_metrics[cat]["token_agreement_pct"].append(agreement)
            category_metrics[cat]["entropy_reduction"].append(entropy_reduction)

            p_label = f"Probe: Loop {probe_pred+1}" if probe_pred >= 0 else ""
            print(f"[{idx:02d}/{len(sample_cases)}] {cat:<30} | CosSim: {cos_sim:.3f} | ||dh||/||h||: {norm_ratio:.3f} | Concordancia: {agreement:4.1f}% | d_Entropia: {entropy_reduction:+.3f} | {p_label}")

    print("\n" + "=" * 75)
    print("SINTESE DA CONTRIBUICAO DOS LOOPS POR CATEGORIA")
    print("=" * 75)
    print(f"{'Categoria':<32} | {'Cos Sim':<8} | {'dh Norm':<8} | {'Concord.':<8} | {'d_Entropia':<10} | {'Decisao Probe'}")
    print("-" * 75)

    summary_report = {}
    for cat, m in category_metrics.items():
        avg_cos = sum(m["cosine_sim"]) / max(1, len(m["cosine_sim"]))
        avg_norm = sum(m["residual_norm_ratio"]) / max(1, len(m["residual_norm_ratio"]))
        avg_agree = sum(m["token_agreement_pct"]) / max(1, len(m["token_agreement_pct"]))
        avg_ent = sum(m["entropy_reduction"]) / max(1, len(m["entropy_reduction"]))
        
        probe_dist = dict(m["probe_decisions"])
        probe_str = ", ".join([f"L{k+1}:{v}" for k, v in sorted(probe_dist.items())])

        print(f"{cat:<32} | {avg_cos:7.3f} | {avg_norm:7.3f} | {avg_agree:6.1f}% | {avg_ent:+9.4f} | {probe_str}")
        summary_report[cat] = {
            "avg_cosine_similarity": avg_cos,
            "avg_residual_norm_ratio": avg_norm,
            "avg_token_agreement_pct": avg_agree,
            "avg_entropy_reduction": avg_ent,
            "probe_decisions": probe_dist
        }

    report_path = os.path.join(ROOT_DIR, "data", "loop_contribution_analysis.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    print(f"\nRelatório completo salvo em: {report_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
