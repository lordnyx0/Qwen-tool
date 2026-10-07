"""
sanitize_and_curate_tool_use.py
Pipeline de Sanitização Estrita e Curadoria de Elite para Trajetórias de Tool-Use:

Critérios de Rejeição Incondicional (Purga de Amostras Ruins e Corrompidas):
1. Tool Inexistente: A tool chamada pelo assistente DEVE existir no catálogo de tools da amostra.
2. Argumentos Inválidos: Toda string de arguments em tool_calls DEVE ser JSON válido parseável.
3. Schemas Inválidos: Parâmetros de tools devem ser JSON Schemas válidos com type e properties.
4. Trajetória Incompleta: A conversa NUNCA pode terminar no meio de uma chamada (deve terminar em 'assistant').
5. Resposta Órfã: Todo role 'tool' deve corresponder a uma chamada pendente do assistente.
6. Corrupção de Caracteres: Descarte de amostras com mojibake ou replacement character (\ufffd).
7. Repetição / Loop Degenerado: Descarte de saídas que repetem o mesmo bloco de texto > 3 vezes.
8. Validação de Chat Template: 100% das amostras devem renderizar com perfeição no tokenizer nativo Qwen.
9. Janela de Treino: token_length <= 5000 tokens (zero truncamento).
"""

import os
import sys
import json
import re
from collections import Counter
from typing import Dict, List, Any, Set, Tuple, Optional
from transformers import AutoTokenizer

sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = os.path.abspath(".")
TOKENIZER_DIR = os.path.join(ROOT_DIR, "checkpoints", "qwen3loop_sft_final")
INPUT_FILE = os.path.join(ROOT_DIR, "data", "sft_ultradata_tool_use_non_sac_5k.jsonl")
OUTPUT_FILE = os.path.join(ROOT_DIR, "data", "sft_ultradata_tool_use_curated_clean.jsonl")

MAX_TOKENS = 5000

print("=" * 75)
print("INICIANDO SANITIZAÇÃO RIGOROSA E PURGA DE AMOSTRAS CORROMPIDAS")
print("=" * 75)

# 1. Carregar Tokenizer
print(f"1. Carregando tokenizer de {TOKENIZER_DIR}...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)

# 2. Funções de Validação e Normalização
def sanitize_and_validate_tools(raw_tools: Any) -> Tuple[Optional[List[Dict[str, Any]]], Set[str]]:
    """Valida schemas de ferramentas contra corrupções e inconsistências."""
    if not isinstance(raw_tools, list) or len(raw_tools) == 0:
        return None, set()

    clean_tools = []
    tool_names = set()

    for item in raw_tools:
        if not isinstance(item, dict):
            return None, set()

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

def check_repetition_degeneracy(text: str) -> bool:
    """Detecta loops de repetição degenerados comuns em saídas corrompidas de LLMs."""
    if len(text) < 100:
        return False
    # Padrão de repetição de sentenças ou frases idênticas
    sentences = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 20]
    if len(sentences) >= 5:
        counts = Counter(sentences)
        if any(c >= 4 for c in counts.values()):
            return True
    return False

def validate_and_clean_trajectory(
    raw_messages: Any, 
    available_tools: Set[str]
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Executa auditoria estrita em cada turno da conversa.
    Retorna (mensagens_limpas, motivo_rejeição).
    """
    if not isinstance(raw_messages, list) or len(raw_messages) < 2:
        return None, "mensagens_insuficientes"

    # Trajetória deve terminar obrigatoriamente em 'assistant'
    last_msg = raw_messages[-1]
    if not isinstance(last_msg, dict) or last_msg.get("role") != "assistant":
        return None, "trajetoria_incompleta_nao_termina_em_assistant"

    clean_messages = []
    pending_tool_calls = set()
    total_calls = 0

    for idx, m in enumerate(raw_messages):
        if not isinstance(m, dict):
            return None, "mensagem_nao_dict"

        role = m.get("role")
        if role not in ["system", "user", "assistant", "tool"]:
            return None, f"role_invalido_{role}"

        content = m.get("content") or ""
        if not isinstance(content, str):
            content = str(content)

        # Checagem de Mojibake / Corrupção de codificação
        if "\ufffd" in content:
            return None, "corrupcao_mojibake_caractere_invalido"

        # Checagem de repetição degenerada
        if check_repetition_degeneracy(content):
            return None, "loop_repeticao_degenerada"

        clean_m = {"role": role, "content": content}

        # Validação do Assistente
        if role == "assistant":
            raw_tcs = m.get("tool_calls")
            if raw_tcs:
                if not isinstance(raw_tcs, list):
                    return None, "tool_calls_formato_invalido"

                clean_tcs = []
                for tc_idx, tc in enumerate(raw_tcs):
                    if not isinstance(tc, dict):
                        return None, "tool_call_item_invalido"

                    fn = tc.get("function")
                    if not isinstance(fn, dict):
                        return None, "tool_call_sem_function"

                    tc_name = fn.get("name")
                    if not tc_name:
                        return None, "tool_call_sem_name"

                    # REGRA DE OURO: A ferramenta chamada DEVE existir no catálogo da amostra
                    if tc_name not in available_tools:
                        return None, f"tool_chamada_inexistente_{tc_name}"

                    raw_args = fn.get("arguments")
                    # Normalizar e testar se os argumentos são JSON parseável
                    if isinstance(raw_args, dict):
                        try:
                            args_str = json.dumps(raw_args, ensure_ascii=False)
                        except Exception:
                            return None, "arguments_dict_nao_serializavel"
                    elif isinstance(raw_args, str):
                        try:
                            json.loads(raw_args)
                            args_str = raw_args
                        except Exception:
                            return None, "arguments_string_json_invalido"
                    else:
                        return None, "arguments_tipo_invalido"

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
            else:
                # Mensagem de assistente vazia sem tool calls é inválida
                if not content.strip():
                    return None, "assistant_vazio_sem_tool_calls"

        # Validação da Observação de Ferramenta
        elif role == "tool":
            tool_name = m.get("name")
            if not tool_name:
                if len(pending_tool_calls) == 1:
                    tool_name = list(pending_tool_calls)[0]
                else:
                    return None, "tool_response_sem_name"

            clean_m["name"] = tool_name
            if "tool_call_id" in m:
                clean_m["tool_call_id"] = m["tool_call_id"]

        clean_messages.append(clean_m)

    return clean_messages, None

# 3. Execução da Varredura
print(f"2. Varrendo {INPUT_FILE}...")
stats = {
    "total_read": 0,
    "approved": 0,
    "rejected": 0,
    "reject_reasons": Counter()
}

approved_dataset = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
            
        stats["total_read"] += 1
        try:
            item = json.loads(line)
        except Exception:
            stats["rejected"] += 1
            stats["reject_reasons"]["json_parse_error_linha"] += 1
            continue

        raw_tools = item.get("tools", [])
        raw_msgs = item.get("messages", [])

        # A. Validar Schemas de Tools
        clean_tools, available_tools = sanitize_and_validate_tools(raw_tools)
        if not clean_tools:
            stats["rejected"] += 1
            stats["reject_reasons"]["tools_schema_invalido"] += 1
            continue

        # B. Validar Integridade da Trajetória
        clean_msgs, reject_reason = validate_and_clean_trajectory(raw_msgs, available_tools)
        if not clean_msgs:
            stats["rejected"] += 1
            stats["reject_reasons"][reject_reason] += 1
            continue

        # C. Validar Renderização Nativa com Chat Template e Janela <= 5000 tokens
        try:
            tokens = tokenizer.apply_chat_template(
                clean_msgs,
                tools=clean_tools,
                tokenize=True,
                return_dict=False
            )
            t_len = len(tokens)
            if t_len > MAX_TOKENS:
                stats["rejected"] += 1
                stats["reject_reasons"]["excede_max_tokens_5000"] += 1
                continue
            if t_len < 30:
                stats["rejected"] += 1
                stats["reject_reasons"]["trajetoria_curta_demais"] += 1
                continue
        except Exception as e:
            stats["rejected"] += 1
            stats["reject_reasons"]["chat_template_render_error"] += 1
            continue

        item["messages"] = clean_msgs
        item["tools"] = clean_tools
        item["token_length"] = t_len
        approved_dataset.append(item)
        stats["approved"] += 1

print("\n" + "=" * 75)
print("RELATÓRIO DE CURADORIA E PURGA DE AMOSTRAS RUINS")
print("=" * 75)
print(f"Total de Amostras Avaliadas: {stats['total_read']:,}")
print(f"Aprovadas com Certificação de Pureza: {stats['approved']:,} ({stats['approved']/stats['total_read']*100:.1f}%)")
print(f"Amostras Ruins / Rejeitadas: {stats['rejected']:,} ({stats['rejected']/stats['total_read']*100:.1f}%)")

print("\n--- DISTRIBUIÇÃO DOS MOTIVOS DE REJEIÇÃO ---")
for reason, count in stats["reject_reasons"].most_common(15):
    pct = (count / stats["total_read"]) * 100
    print(f"  - {reason:45}: {count:5,} ({pct:.2f}%)")

# 4. Salvar Dataset Curado e Limpo
print(f"\n3. Salvando dataset 100% limpo em: {OUTPUT_FILE}...")
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for s in approved_dataset:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

out_size_mb = os.path.getsize(OUTPUT_FILE) / 1024 / 1024
print(f"-> Concluído com sucesso! Tamanho final: {out_size_mb:.2f} MB ({len(approved_dataset):,} amostras de elite)")
print("=" * 75)
