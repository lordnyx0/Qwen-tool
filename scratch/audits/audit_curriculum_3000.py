import sys
import json
import random
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding='utf-8')

INPUT_FILE = "data/cleaned/curriculum_3000_tool_sft.jsonl"

def run_deep_audit():
    print("=" * 80)
    print("AUDITORIA PROFUNDA DE AMOSTRAGEM ALTA - DATASET SFT CURRICULUM (3.000 AMOSTRAS)")
    print("=" * 80)

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    print(f"Total de Amostras Lidas: {len(samples)}")

    pillar_samples = defaultdict(list)
    corrections_by_pillar = defaultdict(list)
    token_lengths_by_pillar = defaultdict(list)

    # 1. Validação Estrutural e de ChatML
    invalid_roles = 0
    missing_tools = 0
    non_assistant_ends = 0
    over_5k = 0

    for idx, s in enumerate(samples):
        pillar = s.get("pillar")
        pillar_samples[pillar].append(s)
        tok_len = s.get("token_length", 0)
        token_lengths_by_pillar[pillar].append(tok_len)
        if tok_len > 5000:
            over_5k += 1

        msgs = s.get("messages", [])
        if not msgs or msgs[-1].get("role") != "assistant":
            non_assistant_ends += 1

        if s.get("is_retrajectory"):
            corrections_by_pillar[pillar].append(s)

        # Checar se ferramentas estão presentes quando chamadas
        has_tc = any("tool_calls" in m for m in msgs)
        if has_tc and not s.get("tools"):
            missing_tools += 1

    print("\n--- [1. VALIDAÇÃO TÉCNICA E CONFORMIDADE] ---")
    print(f"• Amostras com final diferente de assistant: {non_assistant_ends}")
    print(f"• Amostras com tool_calls sem tools declaradas: {missing_tools}")
    print(f"• Amostras > 5.000 tokens (Risco OOM Kaggle T4): {over_5k}")

    print("\n--- [2. DISTRIBUIÇÃO ESTATÍSTICA POR PILAR] ---")
    print(f"{'Pilar':25s} | {'Qtd':5s} | {'Min':5s} | {'Média':6s} | {'Med':5s} | {'Max':5s} | {'Retraj':6s} | {'% Retraj':8s}")
    print("-" * 80)
    for p, items in pillar_samples.items():
        lens = token_lengths_by_pillar[p]
        n_corr = len(corrections_by_pillar[p])
        pct_corr = (n_corr / len(items)) * 100
        print(f"{p:25s} | {len(items):5d} | {min(lens):5d} | {sum(lens)/len(lens):6.1f} | {sorted(lens)[len(lens)//2]:5d} | {max(lens):5d} | {n_corr:6d} | {pct_corr:6.1f}%")

    print("-" * 80)
    total_corr = sum(len(v) for v in corrections_by_pillar.values())
    print(f"{'TOTAL':25s} | {len(samples):5d} | {min(s['token_length'] for s in samples):5d} | {sum(s['token_length'] for s in samples)/len(samples):6.1f} | {sorted(s['token_length'] for s in samples)[len(samples)//2]:5d} | {max(s['token_length'] for s in samples):5d} | {total_corr:6d} | {(total_corr/len(samples))*100:6.1f}%")

    print("\n" + "=" * 80)
    print("--- [3. AMOSTRAGEM REPRESENTATIVA E AUDITORIA MANUAL DE CADA PILAR] ---")
    print("=" * 80)

    for p in ["toolmind_graphsyn", "synth_memory", "nemotron_terminal", "powershell_windows", "hermes_agent_traces", "openclaw_coder"]:
        print(f"\n==================== PILAR: {p.upper()} ====================")
        
        # 1 amostra padrão + 1 amostra de retrajetória
        std_sample = next(x for x in pillar_samples[p] if not x.get("is_retrajectory"))
        ret_sample = next(x for x in pillar_samples[p] if x.get("is_retrajectory"))

        print(f"\n[Amostra Normal]: UUID={std_sample.get('uuid')} | Tokens={std_sample.get('token_length')}")
        print(f"User Prompt: {std_sample['messages'][1]['content'][:180]}...")
        # Achar primeira tool call
        for m in std_sample["messages"]:
            if m.get("tool_calls"):
                arg_val = m['tool_calls'][0]['function']['arguments']
                arg_str = json.dumps(arg_val) if isinstance(arg_val, dict) else str(arg_val)
                print(f"Tool Call: {m['tool_calls'][0]['function']['name']} -> {arg_str[:120]}")
                break

        print(f"\n[Amostra de Retrajetória/Auto-Correção]: UUID={ret_sample.get('uuid')} | Tokens={ret_sample.get('token_length')}")
        for idx, m in enumerate(ret_sample["messages"]):
            role = m["role"]
            content = m.get("content", "")
            tc = m.get("tool_calls")
            if role == "user":
                print(f"  Turn {idx} [USER]: {content[:140]}")
            elif role == "assistant" and tc:
                fn_name = tc[0]["function"]["name"]
                args = tc[0]["function"]["arguments"]
                think = ""
                if "<think>" in content:
                    think = content.split("</think>")[0].replace("<think>", "").strip()[:100] + "..."
                print(f"  Turn {idx} [ASSISTANT]: (Think: {think}) Calling {fn_name}({args[:90]})")
            elif role == "tool":
                print(f"  Turn {idx} [TOOL]: {content[:140]}")
            elif role == "assistant" and not tc:
                print(f"  Turn {idx} [ASSISTANT FINAL]: {content[:140]}")

if __name__ == "__main__":
    run_deep_audit()
