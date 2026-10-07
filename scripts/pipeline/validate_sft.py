"""
validate_sft.py — bateria de gates 04 Passo 2 para qualquer .jsonl SFT.
Uso: python scripts/validate_sft.py <arquivo.jsonl>
Retorna exit 0 se tudo passa; imprime contagens de falha por regra.
"""
import ast
import json
import re
import sys

GENERIC = {"command completed successfully with exit code 0.", "execution succeeded with output.",
           "file content loaded successfully.", '{"status":"ok"}', '{"status": "ok"}'}
MAL = re.compile(r"(mimikatz|sekurlsa|rubeus|kerberoast|tokenmanipulation|defendercontrol|meterpreter|"
                 r"reverse\s*shell|amsi\s*bypass|downloadstring.*(creds|malicious)|malicious_site|"
                 r"s3cur3th1sshit|invoke-mimikatz|send-mailmessage.*t1027)", re.I)
PYPAT = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
MENTIONS = re.compile(r"\b(list_dir|list_directory|read_file|read_text_file|bash|powershell|write_file|search_files|search_code|create_directory|edit_file|move_file)\b")

fp = sys.argv[1]
rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
fails = {"json_lines": 0, "args": 0, "declarada": 0, "fim_assistant": 0,
         "ast": 0, "malicioso": 0, "pii_elena": 0, "generica": 0,
         "desc_decl": 0, "dupname": 0, "reqsubset": 0, "sem_system": 0}
info = {"tc_sem_id": 0, "tool_sem_tcid": 0}  # aviso (staging em voo nao tem ids)
seen = set()
dups = 0
for r in rows:
    if r.get("uuid") in seen:
        dups += 1
    seen.add(r.get("uuid"))
    decl = set(t["function"]["name"] for t in r.get("tools", []))
    if len(decl) != len(r.get("tools", [])):
        fails["dupname"] += 1
    for t in r.get("tools", []):
        fn = t.get("function", {})
        props = set((((fn.get("parameters") or {}).get("properties")) or {}).keys())
        req = (fn.get("parameters") or {}).get("required") or []
        if any(k not in props for k in req):
            fails["reqsubset"] += 1
            break
    for t in r.get("tools", []):
        d = (t.get("function") or {}).get("description", "") or ""
        for m in MENTIONS.findall(d):
            if m != t["function"]["name"] and m not in decl:
                fails["desc_decl"] += 1
                break
    for m in r.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            try:
                json.loads(tc["function"]["arguments"])
            except Exception:
                fails["args"] += 1
            if tc["function"]["name"] not in decl:
                fails["declarada"] += 1
            if not tc.get("id"):
                info["tc_sem_id"] += 1
        if m.get("role") == "tool" and (
                not m.get("content", "").strip() or m.get("content", "").strip() in GENERIC):
            fails["generica"] += 1
        if m.get("role") == "tool" and not m.get("tool_call_id"):
            info["tool_sem_tcid"] += 1
        for b in PYPAT.findall(m.get("content") or ""):
            try:
                ast.parse(b)
            except Exception:
                fails["ast"] += 1
                break
    if not r.get("messages") or r["messages"][-1].get("role") != "assistant":
        fails["fim_assistant"] += 1
    if not r.get("messages") or r["messages"][0].get("role") != "system":
        fails["sem_system"] += 1
    blob = json.dumps(r, ensure_ascii=False)
    if MAL.search(blob):
        fails["malicioso"] += 1
    if "Elena Maria Vasquez" in blob:
        fails["pii_elena"] += 1

print("N:", len(rows), "| uuids duplicados:", dups)
print(fails)
print("info (nao falha):", info)
sys.exit(0 if sum(fails.values()) + dups == 0 else 1)
