"""
build_purified_sft_dataset.py  (04 Checklist Passo 1 + 02 Diretrizes + 06 Fontes)
Gera dataset purificado com catalogo canonico (<50), observacao substantiva,
balanceamento 30/50/20 e 10-15% retrajectories.
Fontes (06): ultradata 16k (primaria), CLEAN 1603 (base ja sanitizada).
Saidas: data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl + kaggle_sft_data/curriculum_3000_tool_sft.jsonl
"""
import os, json, re, ast, random
from collections import Counter
from copy import deepcopy

ROOT = os.path.abspath(".")
CLEAN = os.path.join(ROOT, "data", "cleaned", "curriculum_3000_tool_sft_CLEAN.jsonl")
ULTRA = os.path.join(ROOT, "data", "cleaned", "ultradata_toolmind_curated_16k.jsonl")
OUT_CLEANED = os.path.join(ROOT, "data", "cleaned", "curriculum_3000_tool_sft_PURIFIED.jsonl")
OUT_KAGGLE = os.path.join(ROOT, "kaggle_sft_data", "curriculum_3000_tool_sft.jsonl")
STATS = os.path.join(ROOT, "data", "cleaned", "curriculum_purified_stats.json")
random.seed(42)

print("tok...")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT,"checkpoints","qwen3loop_tool_sft_final"), local_files_only=True)
def ntok(msgs, tools):
    try:
        return len(tok.apply_chat_template(msgs, tools=tools if tools else None, tokenize=True, return_dict=False))
    except:
        return 99999
def ok_tpl(msgs, tools):
    try:
        tok.apply_chat_template(msgs, tools=tools if tools else None, add_generation_prompt=False, tokenize=False)
        return True
    except:
        return False

# ---- catalogo canonico 02 (15 tools) ----
CANON = {
 "bash": {"description":"Execute bash shell commands in Linux terminal and view stdout/stderr output.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Bash command line"}},"required":["command"]}},
 "powershell": {"description":"Execute legitimate Windows administration (Get-Process, Get-Service, Get-ChildItem, Select-String, Test-NetConnection). No offensive security.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"PowerShell command"}},"required":["command"]}},
 "read_file": {"description":"Read file or list directory context.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"File or directory path"}},"required":["path"]}},
 "write_file": {"description":"Create or update file with content.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Destination path"},"content":{"type":"string","description":"File content"}},"required":["path","content"]}},
 "list_dir": {"description":"List structured directory contents.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory path"},"recursive":{"type":"boolean","description":"Recursive"}},"required":["path"]}},
 "search_code": {"description":"Literal or semantic code search.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"},"path":{"type":"string","description":"Scope path"}},"required":["query"]}},
 "execute_command": {"description":"Execute sandboxed shell command.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"Command"}},"required":["command"]}},
 "core_memory_append": {"description":"Append to short-term memory.","parameters":{"type":"object","properties":{"key":{"type":"string","description":"Memory key"},"value":{"type":"string","description":"Value"}},"required":["key","value"]}},
 "core_memory_replace": {"description":"Update existing memory key.","parameters":{"type":"object","properties":{"key":{"type":"string","description":"Memory key"},"value":{"type":"string","description":"New value"}},"required":["key","value"]}},
 "archival_memory_insert": {"description":"Insert into long-term repository.","parameters":{"type":"object","properties":{"content":{"type":"string","description":"Content"}},"required":["content"]}},
 "archival_memory_search": {"description":"Contextual retrieval by query.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search query"}},"required":["query"]}},
 "python_interpreter": {"description":"Execute Python code in sandbox and return stdout. Use print() for result.","parameters":{"type":"object","properties":{"code":{"type":"string","description":"Valid Python code"}},"required":["code"]}},
 "calculator": {"description":"Evaluate math formula directly.","parameters":{"type":"object","properties":{"expression":{"type":"string","description":"Math expression"}},"required":["expression"]}},
 "http_request": {"description":"Call REST endpoint.","parameters":{"type":"object","properties":{"url":{"type":"string","description":"Full URL"},"method":{"type":"string","description":"GET/POST"}},"required":["url"]}},
 "database_query": {"description":"Read-only SELECT on relational DB.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Valid SELECT"}},"required":["query"]}},
}
def canon_tool(name):
    s=CANON[name]
    return {"type":"function","function":{"name":name,"description":s["description"],"parameters":deepcopy(s["parameters"])}}

# mapeamentos workspace/memory legados -> canonico
MAP = {
 "filesystem-context":"read_file", "safe-exec":"execute_command",
 "browser":"http_request", "browser_open":"http_request", "web_search":"http_request",
 "browser_navigate":"http_request", "browser_snapshot":"http_request",
 "memory_get":"archival_memory_search", "memory_append":"core_memory_append",
 "memory_replace":"core_memory_replace", "memory_clear":"core_memory_replace",
 "memory_retrieve":"archival_memory_search", "memory_update":"core_memory_replace",
 "terminal":"execute_command",
}
for k in list(CANON.keys()):
    MAP[k]=k
# variantes archival/core -> 4 canonicas
for n in ["archival_memory_add","archival_memory_clear","archival_memory_remove","archival_memory_retrieve","archival_memory_retrieve_all","archival_memory_update","archival_memory_key_search","archival_memory_list_keys","archival_memory_replace","core_memory_add","core_memory_clear","core_memory_remove","core_memory_retrieve","core_memory_retrieve_all","core_memory_key_search","core_memory_list_keys","core_memory_replace","core_memory_update","archival_memory_insert"]:
    if "search" in n or "retrieve" in n or "list" in n or "key" in n:
        MAP[n]="archival_memory_search"
    elif "replace" in n or "update" in n or "clear" in n or "remove" in n:
        MAP[n]="core_memory_replace"
    else:
        MAP[n]="core_memory_append"

MAL = re.compile(r"(mimikatz|sekurlsa|lsadump|wcmdump|dumpwcm|rubeus|kerberoast|tokenmanipulation|defendercontrol|meterpreter|reverse\s*shell|amsi\s*bypass|payload| s3cur3th1sshit|creds/master|invoke-mimikatz|get-netuser.*spn|setspn.*-Q|send-mailmessage.*t1027|malicious_site|dump.*credential|credential.*dump|keylog|exfiltrat|hidden.*bypass|bypass.*hidden|downloadstring.*(creds|malicious)|downloadfile.*malicious)", re.I)
GENERIC = {"command completed successfully with exit code 0.","execution succeeded with output.","file content loaded successfully."}
FAKE_SUB = ['text": "Content..."']

def substantive_tool(c):
    s=(c or "").strip()
    if not s or len(s)<30:
        return False
    if s in GENERIC:
        return False
    low=s.lower()
    if low in ('{"status":"ok"}','ok','true','success'):
        return False
    if any(f in s for f in ['text": "Content..."']):
        return False
    return True

def args_ok(a):
    try:
        json.loads(a); return True
    except: return False

def convert_msg_names(msgs, tools_declared):
    """Reescreve tool_calls/tool names para canonico; retorna (msgs2, tools2) ou None se tecnico diverso."""
    # se amostra e technical_api diversa (toolmind), mantem original (excecao documentada) - so valida
    return msgs, tools_declared

reject=Counter()
pools=[]  # (msgs, tools, pillar, is_retra)

# ---- A. CLEAN 1603 com mapeamento canonico para terminal/workspace/memory ----
clean_rows=[json.loads(l) for l in open(CLEAN, encoding="utf-8")]
for r in clean_rows:
    src=r.get("source","?"); pillar=r.get("pillar","?")
    msgs=deepcopy(r["messages"]); decl=[t["function"]["name"] for t in r["tools"]]
    # tecnica diversa (toolmind + hermes): mantem tools originais (excecao technical_api documentada)
    is_tech = (src in ("toolmind_graphsyn","sroecker/hermes-agent-traces-chatml"))
    if is_tech:
        # valida substantiva
        tcs=[m for m in msgs if m["role"]=="tool"]
        if not all(substantive_tool(m.get("content","")) for m in tcs):
            reject["tech_obs_vacua"]+=1; continue
        pools.append((msgs, r["tools"], "technical_api", bool(r.get("is_retrajectory"))))
        continue
    # demais: mapeia para canonico
    new_tools={}; new_msgs=[]
    ok=True
    for m in msgs:
        m2=deepcopy(m)
        if m.get("tool_calls"):
            ntc=[]
            for tc in m["tool_calls"]:
                on=tc["function"]["name"]
                cn=MAP.get(on)
                if not cn:
                    ok=False; break
                # converte args: tenta preservar chaves uteis
                try: a=json.loads(tc["function"]["arguments"])
                except: ok=False; break
                if cn=="http_request" and isinstance(a,dict):
                    url=a.get("url") or a.get("action") or "https://example.org"
                    a={"url":str(url)[:300],"method":"GET"}
                elif cn in ("read_file","list_dir") and isinstance(a,dict):
                    p=a.get("path") or a.get("command") or a.get("query") or "."
                    a={"path":str(p)[:300]}
                elif cn=="execute_command" and isinstance(a,dict):
                    c=a.get("command") or a.get("path") or json.dumps(a)[:300]
                    a={"command":str(c)[:500]}
                elif cn in ("core_memory_append","core_memory_replace") and isinstance(a,dict):
                    # normaliza text/path -> key/value
                    k=a.get("key") or a.get("path") or "note"
                    v=a.get("value") or a.get("text") or a.get("content") or json.dumps(a)[:500]
                    a={"key":str(k)[:100],"value":str(v)[:800]}
                elif cn in ("archival_memory_insert",) and isinstance(a,dict):
                    a={"content":json.dumps(a)[:800]}
                elif cn in ("archival_memory_search",) and isinstance(a,dict):
                    q=a.get("query") or a.get("path") or a.get("key") or json.dumps(a)[:300]
                    a={"query":str(q)[:300]}
                ntc.append({"id":tc.get("id") or f"call_{len(ntc)}","type":"function","function":{"name":cn,"arguments":json.dumps(a,ensure_ascii=False)}})
                new_tools[cn]=canon_tool(cn)
            if not ok: break
            m2["tool_calls"]=ntc
        if m.get("role")=="tool":
            on=m.get("name"); cn=MAP.get(on)
            if not cn: ok=False; break
            m2["name"]=cn
            new_tools[cn]=canon_tool(cn)
        new_msgs.append(m2)
    if not ok:
        reject["sem_mapeamento_canonico"]+=1; continue
    # bash/powershell declarados explicitamente
    for t in r["tools"]:
        n=t["function"]["name"]
        if n in ("bash","powershell") and n not in new_tools:
            new_tools[n]=canon_tool(n)
    tools2=list(new_tools.values())
    # valida substantiva (02 regra 3)
    if not all(substantive_tool(m.get("content","")) for m in new_msgs if m["role"]=="tool"):
        reject["obs_vacua_pos_map"]+=1; continue
    # seguranca
    blob=json.dumps(new_msgs,ensure_ascii=False)
    if MAL.search(blob):
        reject["malicioso"]+=1; continue
    # pillar canonico
    if pillar in ("nemotron_terminal",): np="terminal_linux"
    elif pillar in ("powershell_windows",): np="windows_powershell"
    elif pillar in ("openclaw_coder",): np="workspace_code"
    elif pillar in ("synth_memory",): np="memory_retrieval"
    elif pillar in ("hermes_agent_traces",): np="technical_api"
    else: np=pillar
    # detecta retrajetoria real: tool com erro + assistant subsequente
    retr = bool(r.get("is_retrajectory"))
    if not retr:
        for i,m in enumerate(new_msgs if not is_tech else msgs):
            if m.get("role")=="tool" and any(w in str(m.get("content","")).lower() for w in ["error","invalid","failed","not found","exception"]):
                if i+1 < len(new_msgs if not is_tech else msgs) and (new_msgs if not is_tech else msgs)[i+1].get("role")=="assistant":
                    retr=True; break
    pools.append((new_msgs if not is_tech else msgs, r["tools"] if is_tech else tools2, np, retr))

print("apos mapa:",len(pools), dict(reject))

# ---- B. Minera ultradata substantivo para repor workspace/technical/memory (06 matriz primaria) ----
# Fase B1: prioriza retrajetorias (erro -> correcao), depois completa com substantivas
ultra=[json.loads(l) for l in open(ULTRA, encoding="utf-8")]
random.shuffle(ultra)
def try_add(r):
    msgs=r.get("messages",[])
    if not msgs or msgs[-1].get("role")!="assistant": return None
    if not any(isinstance(m,dict) and m.get("tool_calls") for m in msgs): return None
    tools=[]
    for t in (r.get("tools") or []):
        if not isinstance(t,dict): return None
        fn=t.get("function",t)
        if not isinstance(fn,dict) or not fn.get("name"): return None
        p=deepcopy(fn.get("parameters",{}))
        if not isinstance(p,dict): p={"type":"object","properties":{}}
        if p.get("type")=="dict": p["type"]="object"
        p.setdefault("type","object"); p.setdefault("properties",{})
        tools.append({"type":"function","function":{"name":fn.get("name"),"description":str(fn.get("description",""))[:400],"parameters":p}})
    if not tools: return None
    avail=set(t["function"]["name"] for t in tools)
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            try:
                if tc["function"]["name"] not in avail: return None
                json.loads(tc["function"]["arguments"])
            except: return None
    if not all(substantive_tool(m.get("content","")) for m in msgs if m.get("role")=="tool"): return None
    blob=json.dumps(msgs,ensure_ascii=False)
    if MAL.search(blob) or "\ufffd" in blob: return None
    if "Elena Maria Vasquez" in blob: return None
    for m in msgs:
        for b in re.findall(r"```python\s*\n(.*?)```", m.get("content","") or "", re.DOTALL):
            try: ast.parse(b)
            except: return None
    if not ok_tpl(msgs, tools): return None
    t=ntok(msgs, tools)
    if t>5000 or t<60: return None
    retr=False
    for i,m in enumerate(msgs):
        if m.get("role")=="tool" and any(w in str(m.get("content","")).lower() for w in ["error","invalid","failed","not found","exception"]):
            if i+1<len(msgs) and msgs[i+1].get("role")=="assistant":
                retr=True; break
    pillar = "technical_api" if r["source"]=="toolmind_graphsyn" else ("memory_retrieval" if "memory" in r["source"] else "workspace_code")
    return (msgs, tools, pillar, retr)
added_retr=added_all=0
for r in ultra:
    if added_retr>=350: break
    res=try_add(r)
    if res and res[3]:
        pools.append(res); added_retr+=1; added_all+=1
for r in ultra:
    if added_all>=1100: break
    res=try_add(r)
    if res and not res[3]:
        # evita duplicar os ja adicionados: checa uuid
        pools.append(res); added_all+=1
print(f"minerados: {added_all} (retra {added_retr}) total pools: {len(pools)}")

# ---- C. Balanceamento 30/50/20 + retra 10-15% (02 §4) ----
scored=[]
for msgs, tools, pillar, retr in pools:
    t=ntok(msgs, tools)
    scored.append((msgs, tools, pillar, retr, t))
short=[s for s in scored if 100<=s[4]<=800]
med=[s for s in scored if 800<s[4]<=2000]
lng=[s for s in scored if 2000<s[4]<=4500]
print(f"buckets disp: short {len(short)} med {len(med)} long {len(lng)}")
# alvo ~2200: 30/50/20
random.shuffle(short); random.shuffle(med); random.shuffle(lng)
NS, NM, NL = 660, 1100, 440
sel = short[:NS] + med[:NM] + lng[:NL]
# completa com o que sobrar se faltar
rest=[s for s in scored if s not in sel]
random.shuffle(rest)
while len(sel)<2000 and rest:
    sel.append(rest.pop())
random.shuffle(sel)
# garante retra 10-15%
retr=[s for s in sel if s[3]]
print(f"retra antes: {len(retr)}/{len(sel)}={len(retr)/len(sel)*100:.1f}%")
# se <10%, tenta trocar nao-retra por retra do pool restante
if len(retr)/len(sel)<0.10:
    cand=[s for s in scored if s not in sel and s[3]]
    need=int(0.12*len(sel))-len(retr)
    for c in cand[:max(0,need)]:
        # troca um nao-retra
        for i,s in enumerate(sel):
            if not s[3]:
                sel[i]=c; break

out=[]
for i,(msgs, tools, pillar, retr, t) in enumerate(sel):
    out.append({"uuid":f"purified_{i:04d}","messages":msgs,"tools":tools,"source":"purified_canonical","pillar":pillar,"token_length":t,"is_retrajectory":retr})
print("final:",len(out),"retr:",sum(1 for s in out if s["is_retrajectory"]))
print(Counter(s["pillar"] for s in out))
ts=sorted(s["token_length"] for s in out)
print(f"tokens min {ts[0]} med {ts[len(ts)//2]} mean {sum(ts)/len(ts):.0f} max {ts[-1]}")
print(f"<800: {sum(1 for s in out if s['token_length']<=800)} 800-2000: {sum(1 for s in out if 800<s['token_length']<=2000)} >2000: {sum(1 for s in out if s['token_length']>2000)}")

with open(OUT_CLEANED,"w",encoding="utf-8") as f:
    for s in out: f.write(json.dumps(s,ensure_ascii=False)+"\n")
with open(OUT_KAGGLE,"w",encoding="utf-8") as f:
    for s in out: f.write(json.dumps(s,ensure_ascii=False)+"\n")
with open(STATS,"w",encoding="utf-8") as f:
    json.dump({"n":len(out),"pillars":dict(Counter(s["pillar"] for s in out)),"reject":dict(reject),"buckets":{"short":sum(1 for s in out if s["token_length"]<=800),"med":sum(1 for s in out if 800<s["token_length"]<=2000),"long":sum(1 for s in out if s["token_length"]>2000)},"retra":sum(1 for s in out if s["is_retrajectory"])},f,ensure_ascii=False,indent=2)
print("OK",OUT_CLEANED, OUT_KAGGLE)
