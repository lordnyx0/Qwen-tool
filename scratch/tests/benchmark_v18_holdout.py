import os
import sys
import json
import re
import time
import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT_DIR = os.path.abspath(".")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints", "qwen3loop_tool_sft_final")
BATTERY_PATH = os.path.join(ROOT_DIR, "data", "eval_benchmark_battery_v2.jsonl")

# Carregar engine
sys.path.insert(0, os.path.join(ROOT_DIR, "engine"))
import qwen3loop_python
from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM
AutoConfig.register("qwen3loop", Qwen3LoopConfig)
AutoModelForCausalLM.register(Qwen3LoopConfig, Qwen3LoopForCausalLM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print("=" * 80, flush=True)
print("BATERIA DE AVALIAÇÃO HOLDOUT V2 - MODELO QWEN3LOOP TOOL SFT V18", flush=True)
print(f"Dispositivo: {device} ({dtype}) | Checkpoint: {CHECKPOINT_DIR}", flush=True)
print("=" * 80, flush=True)

print("Carregando Tokenizer e Modelo...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_DIR)
config = AutoConfig.from_pretrained(CHECKPOINT_DIR)
model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_DIR,
    config=config,
    torch_dtype=dtype
).to(device)
model.eval()

# Carregar bateria holdout
with open(BATTERY_PATH, "r", encoding="utf-8") as f:
    cases = [json.loads(line) for line in f if line.strip()]

print(f"Total de itens na bateria holdout: {len(cases)}\n", flush=True)

results = []
metrics_by_cat = defaultdict(lambda: {"total": 0, "trigger_ok": 0, "name_ok": 0, "syntax_ok": 0, "args_ok": 0, "success": 0})

def extract_tool_calls(text):
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)
    calls = []
    if not matches and "<tool_call>" in text:
        after = text.split("<tool_call>", 1)[1]
        matches = [after.split("</tool_call>")[0] if "</tool_call>" in after else after]
    for m in matches:
        clean = m.strip()
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, dict) and "name" in parsed:
                calls.append(parsed)
        except Exception:
            pass
    return calls

t0 = time.time()
for idx, case in enumerate(cases):
    cat = case.get("category", "unknown")
    msgs = case.get("messages", [])
    tools = case.get("tools", None)
    expected_tool = case.get("expected_tool")
    expected_args = case.get("expected_args", {})
    
    # Prompt nativo
    prompt_text = tokenizer.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_text = tokenizer.decode(out[0][input_ids.size(1):], skip_special_tokens=False)
    
    # Avaliação
    calls = extract_tool_calls(gen_text)
    
    m = metrics_by_cat[cat]
    m["total"] += 1
    
    is_abstention = (cat == "negative_abstention" or expected_tool is None)
    
    trigger_ok = False
    name_ok = False
    syntax_ok = False
    args_ok = False
    success = False
    
    if is_abstention:
        # Sucesso se NÃO chamou tool_call
        if len(calls) == 0 and "<tool_call>" not in gen_text:
            trigger_ok = True
            name_ok = True
            syntax_ok = True
            args_ok = True
            success = True
    else:
        # Deve ter chamado ferramenta
        if len(calls) > 0:
            trigger_ok = True
            syntax_ok = True
            call = calls[0]
            if call.get("name") == expected_tool:
                name_ok = True
                call_args = call.get("arguments", {})
                if isinstance(call_args, str):
                    try:
                        call_args = json.loads(call_args)
                    except:
                        call_args = {}
                
                # Checar argumentos
                match_args = True
                if expected_args:
                    for k, v in expected_args.items():
                        if str(call_args.get(k)) != str(v):
                            match_args = False
                            break
                if match_args:
                    args_ok = True
                    success = True
                    
    if trigger_ok: m["trigger_ok"] += 1
    if name_ok: m["name_ok"] += 1
    if syntax_ok: m["syntax_ok"] += 1
    if args_ok: m["args_ok"] += 1
    if success: m["success"] += 1
    
    status_icon = "✅" if success else "❌"
    if (idx + 1) % 10 == 0 or (idx + 1) == len(cases):
        total_eval = idx + 1
        curr_succ = sum(v["success"] for v in metrics_by_cat.values())
        print(f"Progresso: [{total_eval:2d}/{len(cases)}] | Sucessos: {curr_succ}/{total_eval} ({curr_succ/total_eval*100:.1f}%) | Tempo: {time.time()-t0:.1f}s", flush=True)

print("\n" + "=" * 80, flush=True)
print("TABELA CONSOLIDADA DE PERFORMANCE: BATERIA HOLDOUT V2", flush=True)
print("=" * 80, flush=True)
print(f"{'Categoria':<32} | {'Amostras':<8} | {'Trigger':<8} | {'Nome Tool':<9} | {'JSON/Args':<9} | {'Acurácia'}", flush=True)
print("-" * 80, flush=True)

total_samples = 0
total_success = 0
for cat, m in metrics_by_cat.items():
    tot = m["total"]
    total_samples += tot
    total_success += m["success"]
    trig_pct = (m["trigger_ok"] / tot) * 100
    name_pct = (m["name_ok"] / tot) * 100
    args_pct = (m["args_ok"] / tot) * 100
    succ_pct = (m["success"] / tot) * 100
    print(f"{cat:<32} | {tot:<8d} | {trig_pct:6.1f}% | {name_pct:7.1f}% | {args_pct:7.1f}% | {succ_pct:6.1f}%", flush=True)

print("-" * 80, flush=True)
overall_acc = (total_success / total_samples) * 100
print(f"{'OVERALL (HOLDOUT BENCHMARK V2)':<32} | {total_samples:<8d} | {'--':<8} | {'--':<9} | {'--':<9} | {overall_acc:6.1f}%", flush=True)
print("=" * 80, flush=True)

# Salvar relatório json
out_json = os.path.join(ROOT_DIR, "data", "benchmark_v18_holdout_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump({
        "overall_accuracy": overall_acc,
        "total_samples": total_samples,
        "total_success": total_success,
        "by_category": dict(metrics_by_cat)
    }, f, indent=2)

print(f"\nRelatório JSON salvo com sucesso em: {out_json}", flush=True)
