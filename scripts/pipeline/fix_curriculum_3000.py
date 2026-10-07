"""
fix_curriculum_3000.py
Corrige todos os vicios detectados em data/cleaned/curriculum_3000_tool_sft.jsonl
para SFT focado em tool-calling no formato nativo Qwen (chat_template.jinja).
"""
import os, json, re, ast, random
from collections import Counter
from copy import deepcopy

ROOT = os.path.abspath(".")
IN = os.path.join(ROOT, "data", "cleaned", "curriculum_3000_tool_sft.jsonl")
OUT = os.path.join(ROOT, "data", "cleaned", "curriculum_3000_tool_sft_CLEAN.jsonl")
STATS_OUT = os.path.join(ROOT, "data", "cleaned", "curriculum_3000_CLEAN_stats.json")

random.seed(1337)

print("Carregando tokenizer...")
try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, "checkpoints", "qwen3loop_tool_sft_final"), local_files_only=True)
    HAS_TOK = True
except Exception as e:
    print(f"WARN tokenizer falhou: {e}, usando fallback")
    tok = None
    HAS_TOK = False

def measure(messages, tools):
    if HAS_TOK:
        try:
            ids = tok.apply_chat_template(messages, tools=tools if tools else None, tokenize=True, return_dict=False)
            return len(ids)
        except Exception:
            pass
    # fallback: ~4 chars per token
    txt = " ".join(str(m.get("content","")) for m in messages) + json.dumps(tools or [])
    return len(txt)//4

def render_ok(messages, tools):
    if not HAS_TOK:
        return True
    try:
        tok.apply_chat_template(messages, tools=tools if tools else None, add_generation_prompt=False, tokenize=False)
        return True
    except Exception as e:
        return False

# ---------- blocklists ----------
MALICIOUS_RE = re.compile(r"(mimikatz|sekurlsa|lsadump|wcmdump|dumpwcm|rubeus|kerberoast|kerberoastable|tokenmanipulation|defendercontrol|defender.*/D\b|s3cur3th1sshit|creds/master|invoke-mimikatz|hash\s*/password|createprocess.*(threadid|processid)|send-mailmessage.*t1027|malicious_site|malicious.*email|dump.*credential|credential.*dump|keylog|reverse.*shell|exfiltrat|bypass.*hidden|hidden.*bypass|executionpolicy\s+bypass.*hidden|downloadstring.*https?://.*(creds|malicious)|downloadfile.*malicious|whitelist.*pentest|t1005|t1027)", re.I)
# fake outputs que indicam alucinacao (openclaw). PS generico mantido se benigno.
FAKE_OUTPUT_SUBSTR = ["text\": \"Content...\"", "https://example.com", "\"Result for ", "{\"status\": \"ok\", \"title\":"]
GENERIC_PS_OK = "Command completed successfully with exit code 0."

OPENCLAW_EXTRA_TOOLS = {
    "browser": {"description": "Navigate browser to URL and snapshot page", "parameters": {"type": "object", "properties": {"action": {"type": "string"}, "url": {"type": "string"}}, "required": []}},
    "web_search": {"description": "Search web for query", "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "count": {"type": "string"}}, "required": ["query"]}},
    "terminal": {"description": "Execute terminal command", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    "write_file": {"description": "Write file content", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path"]}},
    "search_files": {"description": "Search files", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    "read_file": {"description": "Read file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    "todo": {"description": "Manage todo list", "parameters": {"type": "object", "properties": {"action": {"type": "string"}}, "required": []}},
    "execute_code": {"description": "Execute code", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": []}},
}

def fix_tool_schema(raw_tools):
    """type dict->object, garante estrutura OpenAI."""
    if not isinstance(raw_tools, list):
        return None
    out = []
    for t in raw_tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        desc = fn.get("description", "")
        if not isinstance(desc, str):
            desc = str(desc)
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        else:
            params = deepcopy(params)
            if params.get("type") == "dict":
                params["type"] = "object"
            if "type" not in params:
                params["type"] = "object"
            if "properties" not in params:
                params["properties"] = {}
        out.append({"type": "function", "function": {"name": name, "description": desc, "parameters": params}})
    return out if out else None

def parse_args_to_str(raw):
    if isinstance(raw, dict):
        return json.dumps(raw, ensure_ascii=False), True
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return "{}" , True
        try:
            json.loads(s)
            return s, True
        except:
            pass
        # tenta ast.literal_eval para single quotes
        try:
            obj = ast.literal_eval(s)
            return json.dumps(obj, ensure_ascii=False), True
        except:
            return None, False
    return None, False

def convert_hermes(sample):
    """Extrai tools do system, converte <tool_call> -> tool_calls, <tool_response> -> tool."""
    msgs = deepcopy(sample.get("messages", []))
    if not msgs or msgs[0].get("role") != "system":
        return None, "hermes_sem_system"
    sys_text = msgs[0].get("content", "")
    # extrai tools JSON: há um par vazio "<tools> </tools>" no texto instrucional,
    # então testa TODOS os blocos e pega o primeiro que parseia como lista
    declared = None
    for cand in re.findall(r"<tools>(.*?)</tools>", sys_text, re.DOTALL):
        cand = cand.strip()
        if not cand:
            continue
        try:
            obj = json.loads(cand)
            if isinstance(obj, list) and len(obj) > 0:
                declared = obj
                break
        except:
            continue
    if declared is None:
        return None, "hermes_sem_tools_tag"
    tools = fix_tool_schema(declared)
    if not tools:
        return None, "hermes_tools_schema_fail"
    # limpa system: remove bloco <tools> e exemplo <tool_call> template
    clean_sys = re.sub(r"<tools>.*?</tools>", "", sys_text, flags=re.DOTALL)
    # remove exemplo com <function-name>
    clean_sys = re.sub(r"<tool_call>\s*\{\s*\"name\":\s*<function-name>.*?</tool_call>", "", clean_sys, flags=re.DOTALL)
    clean_sys = re.sub(r"\n{3,}", "\n\n", clean_sys).strip()
    if len(clean_sys) < 20:
        clean_sys = "You are a function calling AI model. You may call one or more functions to assist with the user query."
    else:
        # corta para instrucao curta + preserva aviso
        if len(clean_sys) > 600:
            clean_sys = clean_sys[:600]
    new_msgs = [{"role": "system", "content": clean_sys}]
    tool_names = set(t["function"]["name"] for t in tools)
    call_counter = 0
    for mmsg in msgs[1:]:
        role = mmsg.get("role")
        content = mmsg.get("content") or ""
        if role == "user":
            new_msgs.append({"role": "user", "content": content})
        elif role == "assistant":
            # extrai tags (ignora template placeholder)
            tags = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
            tags = [t for t in tags if "<function-name>" not in t]
            if not tags:
                # texto puro
                if not content.strip():
                    continue
                new_msgs.append({"role": "assistant", "content": content})
            else:
                tcs = []
                for t in tags:
                    try:
                        obj = json.loads(t.strip())
                    except:
                        try:
                            obj = ast.literal_eval(t.strip())
                        except:
                            return None, "hermes_call_parse_fail"
                    name = obj.get("name")
                    args = obj.get("arguments", {})
                    if not name or name not in tool_names:
                        return None, f"hermes_call_unknown_{name}"
                    args_s, ok = parse_args_to_str(args if isinstance(args,(dict,str)) else json.dumps(args))
                    if not ok:
                        # tenta se args ja dict com single quotes internas
                        try:
                            args_s = json.dumps(ast.literal_eval(str(args)), ensure_ascii=False)
                        except:
                            return None, "hermes_args_fail"
                    call_counter += 1
                    tcs.append({"id": f"call_h{call_counter}", "type": "function", "function": {"name": name, "arguments": args_s}})
                # remove tags do content, mantem texto residual + think
                residual = re.sub(r"<tool_call>.*?</tool_call>", "", content, flags=re.DOTALL).strip()
                new_msgs.append({"role": "assistant", "content": residual, "tool_calls": tcs})
        elif role == "tool":
            # um tool msg hermes pode conter MULTIPLOS <tool_response> (1 por call paralela) -> split
            responses = re.findall(r"<tool_response>(.*?)</tool_response>", content, re.DOTALL)
            if not responses:
                responses = [content.strip()]
            # coleta ids pendentes do ultimo assistant com calls (em ordem)
            pending_ids = []
            pending_names = []
            for pm in reversed(new_msgs):
                if pm.get("role") == "assistant" and pm.get("tool_calls"):
                    pending_ids = [tc["id"] for tc in pm["tool_calls"]]
                    pending_names = [tc["function"]["name"] for tc in pm["tool_calls"]]
                    break
            for ri, raw in enumerate(responses):
                raw = raw.strip()
                try:
                    obj = json.loads(raw)
                except:
                    try:
                        obj = ast.literal_eval(raw)
                    except:
                        obj = {"content": raw}
                if isinstance(obj, dict) and "name" in obj and "content" in obj:
                    tname = obj["name"]
                    tcont = obj["content"]
                    tcont_s = tcont if isinstance(tcont, str) else json.dumps(tcont, ensure_ascii=False)
                else:
                    tname = pending_names[ri] if ri < len(pending_names) else (pending_names[-1] if pending_names else None)
                    tcont_s = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
                    if not tname:
                        return None, "hermes_tool_sem_nome"
                link_id = pending_ids[ri] if ri < len(pending_ids) else (pending_ids[-1] if pending_ids else f"call_h{call_counter}")
                new_msgs.append({"role": "tool", "name": tname, "tool_call_id": link_id, "content": tcont_s})
        else:
            return None, f"role_invalido_{role}"
    sample2 = deepcopy(sample)
    sample2["messages"] = new_msgs
    sample2["tools"] = tools
    return sample2, None

def anonymize_synth(text):
    # redacao simples de PII de menor: nomes, escola, etc.
    repl = [
        (r"Elena Maria Vasquez", "[USER]"),
        (r"\bElena\b", "[USER]"),
        (r"Maya Chen", "[FRIEND]"),
        (r"Westlake High School", "[SCHOOL]"),
        (r"Austin, Texas", "[CITY]"),
        (r"\bAustin\b", "[CITY]"),
        (r"Dallas", "[CITY2]"),
        (r"Archimedes", "[PET]"),
        (r"Reed or Swarthmore", "[COLLEGE]"),
        (r"\bMIT\b", "[COLLEGE2]"),
        (r"16 years old", "[AGE]"),
        (r"11th grade \(junior\)", "[GRADE]"),
        (r"The Bell Jar.*?Plath", "[BOOK]"),
    ]
    for pat, sub in repl:
        text = re.sub(pat, sub, text)
    return text

# ---------- load ----------
rows = [json.loads(l) for l in open(IN, encoding="utf-8")]
print(f"Input: {len(rows)}")
reject = Counter()
cleaned = []
seen_traj = set()
for r in rows:
    uuid = r.get("uuid", "?")
    src = r.get("source", "?")
    # 0. mojibake fix/drop: substitui, se ainda tiver apos fix e for hermes user com leak, drop? aqui so fixa
    has_bad = any("\ufffd" in str(m.get("content","")) for m in r.get("messages",[]))
    if has_bad:
        # tenta reparar
        for m in r["messages"]:
            if isinstance(m.get("content"), str):
                m["content"] = m["content"].replace("\ufffd", "")
        # se ainda houver problemas de encoding estranho, mantem mas conta
        reject["mojibake_reparado"] += 1  # nao dropa, so repara; drop so se falhar depois

    # 1. hermes conversion
    cur = deepcopy(r)
    if src == "sroecker/hermes-agent-traces-chatml":
        cur, err = convert_hermes(cur)
        if cur is None:
            reject[f"hermes_{err}"] += 1
            continue
    else:
        # fix tools schema
        fixed_tools = fix_tool_schema(cur.get("tools") or [])
        if not fixed_tools:
            reject["tools_schema_invalido"] += 1
            continue
        cur["tools"] = fixed_tools

    # 1b. openclaw: expande declared para incluir chamados reais (browser, web_search...)
    if src == "bellfire/openclaw-coder-dataset":
        declared_names = set(t["function"]["name"] for t in cur["tools"])
        for m in cur.get("messages", []):
            for tc in (m.get("tool_calls") or []):
                try:
                    n = tc["function"]["name"]
                except:
                    continue
                if n not in declared_names and n in OPENCLAW_EXTRA_TOOLS:
                    spec = OPENCLAW_EXTRA_TOOLS[n]
                    cur["tools"].append({"type": "function", "function": {"name": n, "description": spec["description"], "parameters": spec["parameters"]}})
                    declared_names.add(n)

    # 2. normaliza tool_calls args + ids
    avail = set(t["function"]["name"] for t in cur["tools"])
    bad = None
    for idx, m in enumerate(cur["messages"]):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            new_tcs = []
            for j, tc in enumerate(m["tool_calls"]):
                if not isinstance(tc, dict):
                    bad = "tc_nao_dict"; break
                fn = tc.get("function", {})
                if not isinstance(fn, dict):
                    bad = "tc_sem_function"; break
                name = fn.get("name")
                if not name or name not in avail:
                    bad = f"tool_inexistente_{name}"; break
                args_s, ok = parse_args_to_str(fn.get("arguments", "{}"))
                if not ok:
                    bad = "args_json_invalido"; break
                tid = tc.get("id") or f"call_{idx}_{j}"
                new_tcs.append({"id": tid, "type": "function", "function": {"name": name, "arguments": args_s}})
            if bad:
                break
            m["tool_calls"] = new_tcs
            # garante content str
            if not isinstance(m.get("content"), str):
                m["content"] = str(m.get("content") or "")
            # assistant vazio sem tool_calls ja tratado; com tool_calls vazio ok
            if not m["content"].strip() and not new_tcs:
                bad = "assistant_vazio"; break
        elif m.get("role") == "assistant":
            if not isinstance(m.get("content"), str):
                m["content"] = str(m.get("content") or "")
            # remove tool_calls vazio
            if "tool_calls" in m and not m["tool_calls"]:
                del m["tool_calls"]
        elif m.get("role") == "tool":
            if not isinstance(m.get("content"), str):
                m["content"] = json.dumps(m.get("content"), ensure_ascii=False)
            # garante name
            if not m.get("name"):
                bad = "tool_sem_nome"; break
            if not m.get("tool_call_id"):
                # tenta linkar com ultimo assistant
                linked = None
                for pm in reversed(cur["messages"][:idx]):
                    if pm.get("role") == "assistant" and pm.get("tool_calls"):
                        linked = pm["tool_calls"][-1]["id"]
                        break
                if linked:
                    m["tool_call_id"] = linked
                else:
                    bad = "tool_orfa"; break
    if bad:
        reject[bad] += 1
        continue

    # 3. safety
    blob = " ".join([str(m.get("content","")) for m in cur["messages"] if m.get("role") in ("user","assistant")])
    # inclui comandos em tool_calls
    for m in cur["messages"]:
        for tc in (m.get("tool_calls") or []):
            try:
                blob += " " + tc["function"].get("arguments","")
            except:
                pass
    if MALICIOUS_RE.search(blob):
        reject["seguranca_malicioso"] += 1
        continue

    # 4. fake outputs: drop se TODAS tool outputs forem fake openclaw
    tool_contents = [m.get("content","") for m in cur["messages"] if m.get("role")=="tool"]
    if tool_contents:
        fake_all = all(any(f in c for f in FAKE_OUTPUT_SUBSTR) for c in tool_contents)
        if fake_all and src in ("bellfire/openclaw-coder-dataset",):
            reject["fake_placeholder"] += 1
            continue

    # 5. exige >=1 tool_calls (foco tool-calling)
    n_calls = sum(1 for m in cur["messages"] if m.get("tool_calls"))
    if n_calls == 0:
        reject["zero_call"] += 1
        continue
    msgs = cur["messages"]
    has_tool = any(m.get("role") == "tool" for m in msgs)
    if not has_tool:
        # padrao single-turn (Hermes): user -> assistant(calls), sem observacao. Valido.
        # exige que chamadas estejam APENAS no turno final
        early_calls = any(m.get("tool_calls") for m in msgs[:-1])
        if early_calls or msgs[-1].get("role") != "assistant" or not msgs[-1].get("tool_calls"):
            reject["orfao_intermediario_sem_tool"] += 1
            continue
        # ok, mantem como single-turn
    else:
        # multi-turn: trunca orphan final (assistant com calls sem tool depois)
        while len(msgs) >= 2 and msgs[-1].get("role") == "assistant" and msgs[-1].get("tool_calls"):
            msgs = msgs[:-1]
            reject["orfao_final_truncado"] += 1
        if not msgs or msgs[-1].get("role") != "assistant":
            reject["nao_termina_assistant"] += 1
            continue
        if msgs[-1].get("tool_calls"):
            reject["termina_com_call_pendente"] += 1
            continue
        # REPARO: remove tool orfa (tool sem call pendente, bug do conversor Nemotron:
        # assistant texto -> tool). Em vez de dropar amostra, remove o tool stray.
        repaired = []
        pending = 0
        dropped_stray = 0
        for m in msgs:
            if m.get("tool_calls"):
                pending += len(m["tool_calls"])
                repaired.append(m)
            elif m.get("role") == "tool":
                if pending <= 0:
                    dropped_stray += 1
                    continue
                pending -= 1
                repaired.append(m)
            else:
                repaired.append(m)
        if dropped_stray:
            reject["tool_stray_removido"] += dropped_stray
        msgs = repaired
        # apos reparo, exige >=1 ciclo completo restante
        if not any(m.get("tool_calls") for m in msgs):
            reject["zero_call_apos_reparo"] += 1
            continue
        if pending != 0:
            reject["tool_orfa_seq"] += 1
            continue
        # cada assistant com calls deve ser seguido imediatamente por tool (pos-reparo)
        ok_seq = True
        for i, m in enumerate(msgs):
            if m.get("tool_calls"):
                if i+1 >= len(msgs) or msgs[i+1].get("role") != "tool":
                    ok_seq = False
                    break
        if not ok_seq:
            reject["orfao_intermediario"] += 1
            continue
    cur["messages"] = msgs

    # 6. PII synth_memory
    if cur.get("source") == "synth_memory_20260608" or cur.get("pillar") == "synth_memory":
        for m in cur["messages"]:
            if isinstance(m.get("content"), str):
                m["content"] = anonymize_synth(m["content"])
        for t in cur["tools"]:
            try:
                t["function"]["description"] = anonymize_synth(t["function"]["description"])
            except:
                pass

    # 7. repeticao degenerada
    deg = False
    for m in msgs:
        c = m.get("content","")
        if len(c) > 200:
            # 4x mesma frase longa
            parts = [s.strip() for s in re.split(r"[\n\.]{1}", c) if len(s.strip())>30]
            if len(parts) >= 5:
                from collections import Counter as C
                if any(v>=4 for v in C(parts).values()):
                    deg = True
                    break
    if deg:
        reject["repeticao"] += 1
        continue

    # 8. dedup exato
    key = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
    if key in seen_traj:
        reject["duplicado_exato"] += 1
        continue
    seen_traj.add(key)

    # 9. token length + template
    tlen = measure(msgs, cur["tools"])
    if tlen > 5000 or tlen < 30:
        reject[f"token_len_{tlen}"] += 1
        # agrupa
        reject["token_fora_janela"] += 1
        continue
    if not render_ok(msgs, cur["tools"]):
        reject["template_fail"] += 1
        continue
    cur["token_length"] = tlen
    # preserva metadados, limpa campos legados
    cur.pop("domain", None)
    cur.pop("is_sac_policy", None)
    cleaned.append(cur)

print(f"Aprovadas: {len(cleaned)}/{len(rows)}")
print(reject.most_common(30))

# embaralha
random.shuffle(cleaned)
with open(OUT, "w", encoding="utf-8") as f:
    for s in cleaned:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")
stats = {"input": len(rows), "output": len(cleaned), "reject": dict(reject),
         "por_fonte": dict(Counter(s.get("source","?") for s in cleaned)),
         "por_pilar": dict(Counter(s.get("pillar","?") for s in cleaned))}
with open(STATS_OUT, "w", encoding="utf-8") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(f"Salvo {OUT}")
print(stats)
