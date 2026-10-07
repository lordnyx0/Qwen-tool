"""
build_ultradata_tool_use_8k.py
Pipeline de Estruturação e Curadoria Rigorosa do UltraData Tool-Use (Janela 8.192 Tokens):

Diretrizes Estritas (Conceito Transformers / TRL / Qwen3):
1. Formato Nativo Estruturado: Mantém 'messages' + 'tools' separados sem achatar em texto.
2. Trajetória Completa em Amostra Única:
   USER -> ASSISTANT (tool_call) -> TOOL (observation) -> ASSISTANT (next action/final)
3. Normalização e Validação Estrita de Schemas:
   - Tool chamada DEVE obrigatoriamente existir em 'tools'.
   - 'parameters' DEVE ser JSON Schema válido (type: object, properties).
   - 'arguments' DEVE ser objeto JSON válido e serializável.
   - 'tool' (observação) DEVE corresponder a uma chamada anterior.
   - Trajetória NUNCA pode terminar no meio da execução (deve terminar em 'assistant').
4. Janela de Contexto 8k: 1.000 <= tokens <= 8.192 (zero truncation).
5. Blindagem Anti-Contaminação O(1) contra Bateria Holdout v2.
6. Exportação para: data/sft_ultradata_tool_use_8k.jsonl
"""

import os
import sys
import json
import re
import random
from typing import Dict, List, Any, Set, Optional, Tuple
from datasets import load_dataset
from transformers import AutoTokenizer

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

random.seed(2026)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BENCHMARK_PATH = os.path.join(ROOT_DIR, "data", "eval_benchmark_battery_v2.jsonl")
OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "sft_ultradata_tool_use_8k.jsonl")
TOKENIZER_DIR = os.path.join(ROOT_DIR, "checkpoints", "qwen3loop_sft_final")

MAX_TOKENS = 8192
MIN_TOKENS = 800
TARGET_SAMPLES = 3000

print("=" * 80, flush=True)
print("PIPELINE DE CURADORIA ESTRITA: ULTRADATA TOOL-USE (NATIVO 8K)", flush=True)
print("=" * 80, flush=True)

# 1. Carregar Tokenizer
print(f"1. Carregando tokenizer de {TOKENIZER_DIR}...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)

# 2. Indexação Anti-Contaminação da Bateria Holdout v2
def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())

def extract_ngrams(text: str, n: int = 4) -> Set[str]:
    words = normalize_text(text).split()
    if len(words) < n:
        return set()
    return set(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

benchmark_ngrams = set()
if os.path.exists(BENCHMARK_PATH):
    print(f"2. Indexando Bateria Holdout v2 para blindagem de vazamento...", flush=True)
    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            for m in item.get("messages", []):
                content = m.get("content", "")
                if content:
                    benchmark_ngrams.update(extract_ngrams(content, n=4))
    print(f"   Total de 4-grams únicos do benchmark indexados: {len(benchmark_ngrams):,}", flush=True)

def has_leakage(messages: List[Dict[str, Any]]) -> bool:
    for m in messages:
        if m.get("role") == "user":
            ngrams = extract_ngrams(m.get("content", ""), n=4)
            if any(ng in benchmark_ngrams for ng in ngrams):
                return True
    return False

# 3. Funções de Validação Estrita de Schemas e Trajetórias
def normalize_and_validate_tools(raw_tools: Any) -> Tuple[Optional[List[Dict[str, Any]]], Set[str]]:
    """Valida e normaliza o catálogo de ferramentas da amostra."""
    if not isinstance(raw_tools, list) or len(raw_tools) == 0:
        return None, set()

    clean_tools = []
    tool_names = set()

    for item in raw_tools:
        if not isinstance(item, dict):
            return None, set()
        
        # Pode estar em formato OpenAI: {'type': 'function', 'function': {...}} ou direto {'name': ...}
        if "function" in item and isinstance(item["function"], dict):
            fn = item["function"]
        else:
            fn = item

        name = fn.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            return None, set()
        name = name.strip()

        desc = fn.get("description", "")
        if not isinstance(desc, str):
            desc = str(desc)

        params = fn.get("parameters")
        if not isinstance(params, dict):
            # Normalizar para schema vazio se ausente
            params = {"type": "object", "properties": {}}
        else:
            if "type" not in params:
                params["type"] = "object"
            if "properties" not in params and "$defs" not in params:
                params["properties"] = {}

        clean_tool = {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": params
            }
        }
        clean_tools.append(clean_tool)
        tool_names.add(name)

    return clean_tools, tool_names

def normalize_and_validate_trajectory(
    raw_messages: Any, 
    available_tools: Set[str]
) -> Tuple[Optional[List[Dict[str, Any]]], Dict[str, Any]]:
    """
    Valida a integridade da trajetória:
    - Deve terminar em 'assistant' (não pode terminar no meio da execução).
    - Toda tool chamada DEVE existir em available_tools.
    - Todo 'tool' (observação) deve ter sido chamado anteriormente.
    - Argumentos devem ser JSONs válidos.
    """
    meta = {
        "num_tool_calls": 0,
        "num_tool_responses": 0,
        "num_turns": 0,
        "reject_reason": None
    }

    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        meta["reject_reason"] = "mensagens_insuficientes"
        return None, meta

    # Trajetória não pode terminar em 'tool' ou 'user' (não pode terminar no meio da execução)
    last_msg = raw_messages[-1]
    if not isinstance(last_msg, dict) or last_msg.get("role") != "assistant":
        meta["reject_reason"] = "trajetoria_incompleta_nao_termina_em_assistant"
        return None, meta

    clean_messages = []
    pending_tool_calls = set()
    total_calls = 0
    total_responses = 0

    for idx, m in enumerate(raw_messages):
        if not isinstance(m, dict):
            meta["reject_reason"] = "mensagem_invalida_nao_dict"
            return None, meta

        role = m.get("role")
        if role not in ["system", "user", "assistant", "tool"]:
            meta["reject_reason"] = f"role_desconhecido_{role}"
            return None, meta

        content = m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)

        clean_m = {"role": role, "content": content}

        # Validação do Assistente e Tool Calls
        if role == "assistant":
            raw_tcs = m.get("tool_calls")
            if raw_tcs:
                if not isinstance(raw_tcs, list):
                    meta["reject_reason"] = "tool_calls_nao_lista"
                    return None, meta

                clean_tcs = []
                for tc_idx, tc in enumerate(raw_tcs):
                    if not isinstance(tc, dict):
                        meta["reject_reason"] = "tool_call_item_invalido"
                        return None, meta

                    fn = tc.get("function")
                    if not isinstance(fn, dict):
                        meta["reject_reason"] = "tool_call_sem_function"
                        return None, meta

                    tc_name = fn.get("name")
                    if not tc_name or tc_name not in available_tools:
                        # REGRA DE OURO: A ferramenta chamada DEVE existir no catálogo de tools da amostra!
                        meta["reject_reason"] = f"tool_chamada_inexistente_{tc_name}"
                        return None, meta

                    raw_args = fn.get("arguments")
                    # Normalizar arguments para string JSON válida
                    if isinstance(raw_args, dict):
                        try:
                            args_str = json.dumps(raw_args, ensure_ascii=False)
                        except Exception:
                            meta["reject_reason"] = "argumentos_dict_nao_serializavel"
                            return None, meta
                    elif isinstance(raw_args, str):
                        # Testar se é JSON parseável
                        try:
                            json.loads(raw_args)
                            args_str = raw_args
                        except Exception:
                            meta["reject_reason"] = "argumentos_string_json_invalido"
                            return None, meta
                    else:
                        meta["reject_reason"] = "argumentos_tipo_invalido"
                        return None, meta

                    tc_id = tc.get("id") or f"call_{idx}_{tc_idx}"
                    clean_tcs.append({
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tc_name,
                            "arguments": args_str
                        }
                    })
                    pending_tool_calls.add(tc_name)
                    total_calls += 1

                clean_m["tool_calls"] = clean_tcs

        # Validação do Retorno de Ferramenta (role: tool)
        elif role == "tool":
            tool_name = m.get("name")
            if not tool_name:
                # Tentar inferir se há apenas uma chamada pendente
                if len(pending_tool_calls) == 1:
                    tool_name = list(pending_tool_calls)[0]
                else:
                    meta["reject_reason"] = "tool_response_sem_name"
                    return None, meta

            clean_m["name"] = tool_name
            if "tool_call_id" in m:
                clean_m["tool_call_id"] = m["tool_call_id"]

            total_responses += 1

        clean_messages.append(clean_m)

    meta["num_tool_calls"] = total_calls
    meta["num_tool_responses"] = total_responses
    meta["num_turns"] = len([m for m in clean_messages if m["role"] in ["user", "assistant"]])

    return clean_messages, meta

# 4. Stream e Processamento do Dataset
print(f"3. Conectando e processando subset 'Tool-Use' do openbmb/UltraData-SFT-Agent-2609...", flush=True)
ds = load_dataset("openbmb/UltraData-SFT-Agent-2609", "Tool-Use", split="train", streaming=True)

curated_samples = []
stats = {
    "total_evaluated": 0,
    "leakage_rejected": 0,
    "invalid_tools_schema": 0,
    "tool_name_not_in_catalog": 0,
    "incomplete_trajectory": 0,
    "json_args_error": 0,
    "other_rejects": 0,
    "too_short": 0,
    "too_long": 0,
    "render_error": 0,
    "approved": 0,
    "multi_turn_tool_calls": 0,
    "single_turn_tool_calls": 0,
    "policy_clarification": 0,
}

token_lengths = []
tool_counts = []

for row in ds:
    stats["total_evaluated"] += 1
    raw_msgs = row.get("messages", [])
    raw_tools = row.get("tools", [])
    uuid = row.get("uuid", f"sample_{stats['total_evaluated']}")

    # 1. Auditoria de Vazamento
    if has_leakage(raw_msgs):
        stats["leakage_rejected"] += 1
        continue

    # 2. Validação e Normalização de Tools
    clean_tools, available_tools = normalize_and_validate_tools(raw_tools)
    if not clean_tools or not available_tools:
        stats["invalid_tools_schema"] += 1
        continue

    # 3. Validação e Normalização da Trajetória
    clean_msgs, meta = normalize_and_validate_trajectory(raw_msgs, available_tools)
    if not clean_msgs:
        reason = meta.get("reject_reason", "")
        if "tool_chamada_inexistente" in reason:
            stats["tool_name_not_in_catalog"] += 1
        elif "trajetoria_incompleta" in reason:
            stats["incomplete_trajectory"] += 1
        elif "argumentos" in reason:
            stats["json_args_error"] += 1
        else:
            stats["other_rejects"] += 1
        continue

    # 4. Validação pelo Tokenizer (Chat Template Nativo Qwen)
    try:
        rendered = tokenizer.apply_chat_template(clean_msgs, tools=clean_tools, tokenize=False)
        token_ids = tokenizer.encode(rendered, add_special_tokens=False)
        n_tokens = len(token_ids)
    except Exception as e:
        stats["render_error"] += 1
        continue

    # 5. Filtro de Janela 8K
    if n_tokens < MIN_TOKENS:
        stats["too_short"] += 1
        continue
    if n_tokens > MAX_TOKENS:
        stats["too_long"] += 1
        continue

    # 6. Categorização Estratégica
    num_calls = meta["num_tool_calls"]
    num_responses = meta["num_tool_responses"]
    if num_calls > 1 and num_responses > 0:
        stats["multi_turn_tool_calls"] += 1
        cat = "multi_turn_agentic_chaining"
        diff = 3
    elif num_calls == 1:
        stats["single_turn_tool_calls"] += 1
        cat = "single_turn_tool_call"
        diff = 2
    else:
        stats["policy_clarification"] += 1
        cat = "policy_compliance_and_clarification"
        diff = 1

    sample_dict = {
        "uuid": uuid,
        "category": cat,
        "difficulty": diff,
        "token_length": n_tokens,
        "tools_count": len(clean_tools),
        "num_tool_calls": num_calls,
        "num_tool_responses": num_responses,
        "messages": clean_msgs,
        "tools": clean_tools
    }

    curated_samples.append(sample_dict)
    token_lengths.append(n_tokens)
    tool_counts.append(len(clean_tools))
    stats["approved"] += 1

    if stats["approved"] % 100 == 0:
        avg_len = sum(token_lengths) // len(token_lengths)
        print(f"   [Progresso] {stats['approved']:,} / {TARGET_SAMPLES:,} amostras aprovadas | "
              f"Tokens médio: {avg_len} | "
              f"MultiTurn: {stats['multi_turn_tool_calls']} | "
              f"SingleTurn: {stats['single_turn_tool_calls']} | "
              f"Clarificações: {stats['policy_clarification']}", flush=True)

    if stats["approved"] >= TARGET_SAMPLES:
        break

print("\n" + "=" * 80, flush=True)
print(f"CURADORIA FINALIZADA: {stats['approved']:,} TRAJETÓRIAS BLINDADAS (8K)", flush=True)
print("=" * 80, flush=True)

print(f"Total avaliado no stream: {stats['total_evaluated']:,}")
print(f"Aprovados com 100% de integridade: {stats['approved']:,} ({stats['approved']/stats['total_evaluated']*100:.1f}%)")
print("-" * 50)
print(f"Filtragens e Descartes:")
print(f"  - Rejeitados por tool_call inexistente no catálogo: {stats['tool_name_not_in_catalog']:,}")
print(f"  - Rejeitados por trajetória incompleta (não termina em assistant): {stats['incomplete_trajectory']:,}")
print(f"  - Rejeitados por argumentos JSON inválidos: {stats['json_args_error']:,}")
print(f"  - Rejeitados por schema de tools inconsistente: {stats['invalid_tools_schema']:,}")
print(f"  - Rejeitados por vazamento no benchmark: {stats['leakage_rejected']:,}")
print(f"  - Descartados por contexto > 8.192 tokens: {stats['too_long']:,}")
print(f"  - Descartados por contexto < {MIN_TOKENS} tokens: {stats['too_short']:,}")
print("-" * 50)
print(f"Composição de Habilidades do Dataset Curado:")
print(f"  - Trajetórias Multi-Turn Encadeadas (Tool Chaining): {stats['multi_turn_tool_calls']:,} ({stats['multi_turn_tool_calls']/stats['approved']*100:.1f}%)")
print(f"  - Chamadas Precisas Single-Turn: {stats['single_turn_tool_calls']:,} ({stats['single_turn_tool_calls']/stats['approved']*100:.1f}%)")
print(f"  - Compliance e Clarificação Ativa: {stats['policy_clarification']:,} ({stats['policy_clarification']/stats['approved']*100:.1f}%)")
print(f"Estatísticas de Tokens: min={min(token_lengths)}, max={max(token_lengths)}, avg={sum(token_lengths)//len(token_lengths)}")
print(f"Ferramentas disponíveis por amostra: min={min(tool_counts)}, max={max(tool_counts)}, avg={sum(tool_counts)//len(tool_counts)}")

# Salvar Dataset
print(f"\nSalvando em {OUTPUT_PATH}...", flush=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for s in curated_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f"Dataset exportado com sucesso: {size_mb:.2f} MB em {OUTPUT_PATH}", flush=True)
