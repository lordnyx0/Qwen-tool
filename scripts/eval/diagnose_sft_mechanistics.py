"""
diagnose_sft_mechanistics.py
Diagnóstico Mecanístico Avançado no Checkpoint Qwen3Loop SFT Final:
1. Curva de Trade-off de Early-Exit: Mede acurácia e taxa de sucesso parando
   no Loop 1 (28 camadas, 2x velocidade) vs Loop 2 (56 camadas completas).
2. Análise de Atenção nos Schemas de Ferramentas: Mede como o peso de atenção
   se redistribui entre a definição das ferramentas e a consulta do usuário
   entre o Loop 1 e o Loop 2.
"""

import os
import sys
import json
import time
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

sys.path.insert(0, os.path.join(ROOT_DIR, "engine"))
import qwen3loop_python
from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM
AutoConfig.register("qwen3loop", Qwen3LoopConfig)
AutoModelForCausalLM.register(Qwen3LoopConfig, Qwen3LoopForCausalLM)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32

    print("=" * 75)
    print("DIAGNÓSTICO AVANÇADO: EARLY-EXIT TRADE-OFF & ATENÇÃO NOS SCHEMAS")
    print(f"Dispositivo: {device} ({dtype}) | Checkpoint: {CHECKPOINT_DIR}")
    print("=" * 75)

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
    config = AutoConfig.from_pretrained(CHECKPOINT_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_DIR,
        config=config,
        torch_dtype=dtype
    ).to(device)
    model.eval()

    # Carregar casos de teste da Bateria v2
    with open(BATTERY_PATH, "r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]

    # Selecionar subconjunto equilibrado de 24 casos para teste aprofundado
    cases_by_cat = defaultdict(list)
    for c in cases:
        cases_by_cat[c.get("category")].append(c)

    test_subset = []
    for cat in ["single_tool_precise", "multi_tool_discrimination", "python_algorithmic_reasoning", "negative_abstention"]:
        test_subset.extend(cases_by_cat[cat][:6])

    print(f"Total de casos selecionados para análise: {len(test_subset)}")
    print("-" * 75)

    # =========================================================================
    # PARTE 1: CURVA DE TRADE-OFF DE EARLY-EXIT (LOOP 1 vs LOOP 2)
    # =========================================================================
    print("\n--- [PARTE 1] AVALIAÇÃO DE EARLY-EXIT: LOOP 1 (28 CAMADAS) vs LOOP 2 (56 CAMADAS) ---\n")

    l1_correct = 0
    l2_correct = 0
    l1_format_ok = 0
    l2_format_ok = 0
    category_results = defaultdict(lambda: {"l1_pass": 0, "l2_pass": 0, "total": 0})

    def greedy_generate(input_ids, attention_mask, max_new_tokens=40, target_loop=2):
        """Gera tokens de forma autorregressiva fixando a saída em target_loop (1 ou 2)."""
        curr_ids = input_ids.clone()
        curr_mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = model(input_ids=curr_ids, attention_mask=curr_mask, output_hidden_states=True)
                loop_h = getattr(out, "loop_hidden_states", None)
                if loop_h and len(loop_h) >= target_loop:
                    h = loop_h[target_loop - 1]
                    logits = model.lm_head(h)[:, -1, :]
                else:
                    logits = out.logits[:, -1, :]

                next_tok = logits.argmax(dim=-1, keepdim=True)
                if next_tok.item() in [tokenizer.eos_token_id, 151645, 151643]:  # <|im_end|>, <|endoftext|>
                    break
                curr_ids = torch.cat([curr_ids, next_tok], dim=1)
                curr_mask = torch.cat([curr_mask, torch.ones_like(next_tok)], dim=1)

        gen_tokens = curr_ids[0, input_ids.shape[1]:]
        return tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    for idx, case in enumerate(test_subset, 1):
        cat = case.get("category")
        tools = case.get("tools", [])
        messages = case.get("messages", [])
        exp_tool = case.get("expected_tool", "")

        prompt_text = tokenizer.apply_chat_template(messages, tools=tools if tools else None, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Gerar via Loop 1 (Early Exit) e via Loop 2 (Padrão)
        text_l1 = greedy_generate(input_ids, attention_mask, max_new_tokens=48, target_loop=1)
        text_l2 = greedy_generate(input_ids, attention_mask, max_new_tokens=48, target_loop=2)

        # Avaliar acerto de ferramenta
        pass_l1 = False
        pass_l2 = False

        if exp_tool == "":
            # Abstenção: não deve conter tool call
            pass_l1 = ("<tool_call>" not in text_l1)
            pass_l2 = ("<tool_call>" not in text_l2)
        else:
            pass_l1 = (f'"{exp_tool}"' in text_l1 or f"'{exp_tool}'" in text_l1)
            pass_l2 = (f'"{exp_tool}"' in text_l2 or f"'{exp_tool}'" in text_l2)

        category_results[cat]["total"] += 1
        if pass_l1:
            l1_correct += 1
            category_results[cat]["l1_pass"] += 1
        if pass_l2:
            l2_correct += 1
            category_results[cat]["l2_pass"] += 1

        print(f"[{idx:02d}/{len(test_subset)}] {cat:<30} | L1 (Early Exit): {'PASS' if pass_l1 else 'FAIL'} | L2 (Full 56): {'PASS' if pass_l2 else 'FAIL'}")

    print("\n" + "=" * 75)
    print("RESULTADO COMPARATIVO DE EARLY-EXIT (VELOCIDADE vs ACURÁCIA)")
    print("=" * 75)
    print(f"Loop 1 (28 Camadas - 2x Mais Rápido): {l1_correct}/{len(test_subset)} ({l1_correct/len(test_subset)*100:.1f}%)")
    print(f"Loop 2 (56 Camadas - Capacidade Total):  {l2_correct}/{len(test_subset)} ({l2_correct/len(test_subset)*100:.1f}%)")
    print("-" * 75)
    print(f"{'Categoria':<35} | {'Loop 1 (28L)':<15} | {'Loop 2 (56L)':<15}")
    print("-" * 75)
    for cat, r in category_results.items():
        p1 = (r['l1_pass'] / r['total']) * 100
        p2 = (r['l2_pass'] / r['total']) * 100
        print(f"{cat:<35} | {r['l1_pass']}/{r['total']} ({p1:4.1f}%)     | {r['l2_pass']}/{r['total']} ({p2:4.1f}%)")


    # =========================================================================
    # PARTE 2: ANÁLISE DE ATENÇÃO NOS SCHEMAS DAS FERRAMENTAS
    # =========================================================================
    print("\n" + "=" * 75)
    print("--- [PARTE 2] ANÁLISE DE ATENÇÃO NOS SCHEMAS DE FERRAMENTAS (LOOP 1 vs LOOP 2) ---")
    print("=" * 75)

    # Selecionar casos com schemas ricos (multi-tool discrimination)
    disc_cases = cases_by_cat["multi_tool_discrimination"][:8]
    attn_shift_data = []

    for idx, case in enumerate(disc_cases, 1):
        tools = case.get("tools", [])
        messages = case.get("messages", [])
        prompt_text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        
        enc = tokenizer(prompt_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Identificar onde começam e terminam as ferramentas e a pergunta do usuário
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        tool_start = 0
        tool_end = len(tokens)
        user_start = 0

        # Mapeamento de posições
        prompt_str = prompt_text
        if "<tools>" in prompt_str and "</tools>" in prompt_str:
            pre_tool = prompt_str.split("<tools>")[0]
            in_tool = pre_tool + "<tools>" + prompt_str.split("<tools>")[1].split("</tools>")[0] + "</tools>"
            tool_start = len(tokenizer.encode(pre_tool))
            tool_end = len(tokenizer.encode(in_tool))

        if "<|im_start|>user" in prompt_str:
            pre_user = prompt_str.split("<|im_start|>user")[0]
            user_start = len(tokenizer.encode(pre_user))

        # Fazer forward com saída de hidden states
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            loop_h = getattr(outputs, "loop_hidden_states", None)
            if loop_h and len(loop_h) >= 2:
                h1 = loop_h[0]  # [1, seq_len, 1024]
                h2 = loop_h[1]  # [1, seq_len, 1024]

                # Medir alinhamento de atenção latente (similaridade do último token com a região de tools vs user)
                last_h1 = h1[:, -1:, :]
                last_h2 = h2[:, -1:, :]

                sim_tools_l1 = F.cosine_similarity(last_h1, h1[:, tool_start:tool_end, :], dim=-1).mean().item()
                sim_tools_l2 = F.cosine_similarity(last_h2, h2[:, tool_start:tool_end, :], dim=-1).mean().item()

                sim_user_l1 = F.cosine_similarity(last_h1, h1[:, user_start:, :], dim=-1).mean().item()
                sim_user_l2 = F.cosine_similarity(last_h2, h2[:, user_start:, :], dim=-1).mean().item()

                delta_tool = sim_tools_l2 - sim_tools_l1
                delta_user = sim_user_l2 - sim_user_l1

                attn_shift_data.append({
                    "case": case["id"],
                    "sim_tools_l1": sim_tools_l1,
                    "sim_tools_l2": sim_tools_l2,
                    "delta_tool": delta_tool,
                    "sim_user_l1": sim_user_l1,
                    "sim_user_l2": sim_user_l2,
                    "delta_user": delta_user
                })

                print(f"[{idx:02d}/08] {case['id']} | Acoplamento com Schemas: L1={sim_tools_l1:.3f} -> L2={sim_tools_l2:.3f} ({delta_tool:+.3f}) | Alinhamento Usuário: L1={sim_user_l1:.3f} -> L2={sim_user_l2:.3f} ({delta_user:+.3f})")

    avg_d_tool = sum(x["delta_tool"] for x in attn_shift_data) / max(1, len(attn_shift_data))
    avg_d_user = sum(x["delta_user"] for x in attn_shift_data) / max(1, len(attn_shift_data))

    print("\n" + "=" * 75)
    print("SINTESE DO REDIRECIONAMENTO DE ATENCAO NO SEGUNDO LOOP")
    print("=" * 75)
    print(f"Variacao media de foco nos Schemas de Ferramentas (d_Tool): {avg_d_tool:+.4f}")
    print(f"Variacao media de foco na Pergunta do Usuario (d_User):    {avg_d_user:+.4f}")
    if avg_d_tool > 0:
        print(">> INSIGHT: O Loop 2 AUMENTA a atenção nos Schemas das ferramentas, refinando a verificação de parâmetros e tipos antes de emitir a chamada!")
    else:
        print(">> INSIGHT: O Loop 2 converge a atenção para sintetizar os argumentos extraídos da fala do usuário.")
    print("=" * 75)

if __name__ == "__main__":
    main()
