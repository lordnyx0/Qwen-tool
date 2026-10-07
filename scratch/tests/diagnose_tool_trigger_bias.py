import json
import os
import sys
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, "engine")
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import qwen3loop_python

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_dir = "checkpoints/qwen3loop_tool_sft_final"

print("Carregando modelo para diagnóstico de comportamento...", flush=True)
tok = AutoTokenizer.from_pretrained(checkpoint_dir)
cfg = AutoConfig.from_pretrained(checkpoint_dir)
model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, config=cfg, torch_dtype=torch.float16).to(device)
model.eval()

# Definir ferramentas de teste
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": "Get current stock price for a ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "track_package",
            "description": "Track a shipment by tracking number.",
            "parameters": {
                "type": "object",
                "properties": {"tracking_number": {"type": "string"}},
                "required": ["tracking_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "restart_server",
            "description": "Restart a server by server_id. Requires user authentication.",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string"},
                    "user_email": {"type": "string"}
                },
                "required": ["server_id", "user_email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_tax",
            "description": "Calculate sales tax for an amount and zip code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "zip_code": {"type": "string"}
                },
                "required": ["amount", "zip_code"]
            }
        }
    }
]

test_cases = [
    {
        "title": "Caso 1: Consulta de Ações (Leitura pura / Sem autenticação)",
        "messages": [
            {"role": "user", "content": "What is the current stock price of Apple (AAPL)?"}
        ]
    },
    {
        "title": "Caso 2: Rastreio de Pacote com Código Fornecido",
        "messages": [
            {"role": "user", "content": "Please track my package with tracking number TRK-98765."}
        ]
    },
    {
        "title": "Caso 3: Comando com Credenciais Já Fornecidas no Prompt",
        "messages": [
            {"role": "user", "content": "My email is admin@company.com. Please restart server srv-prod-01."}
        ]
    },
    {
        "title": "Caso 4: Diálogo Multi-Turno (Usuário acabou de fornecer o e-mail solicitado)",
        "messages": [
            {"role": "user", "content": "Can you restart server srv-prod-01?"},
            {"role": "assistant", "content": "I'd be happy to help! Could you please verify your user email first?"},
            {"role": "user", "content": "Sure, my email is ops@cloud.io. Please go ahead and restart it."}
        ]
    },
    {
        "title": "Caso 5: Cálculo Matemático Direto",
        "messages": [
            {"role": "user", "content": "Calculate the sales tax for an order of $150.00 in zip code 90210."}
        ]
    }
]

print("\n" + "="*80, flush=True)
print("TESTES DE COMPORTAMENTO: IDENTIFICANDO O VÍCIO DE VERIFICAÇÃO", flush=True)
print("="*80, flush=True)

for tc in test_cases:
    print(f"\n--- {tc['title']} ---", flush=True)
    prompt = tok.apply_chat_template(tc["messages"], tools=tools, add_generation_prompt=True, tokenize=False)
    ids = tok.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=160,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )
    gen = tok.decode(out[0][ids.size(1):], skip_special_tokens=False)
    
    has_tool_call = "<tool_call>" in gen
    print(f"Disparou Tool Call Imediato? {'SIM ✅' if has_tool_call else 'NÃO ⚠️ (Texto/Clarificação)'}")
    print(f"Resposta Gerada:\n{gen.strip()}\n", flush=True)
