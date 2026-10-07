"""validate_calls_schema.py - calls contra defs via jsonschema (best-effort).
Uso: python scripts/pipeline/validate_calls_schema.py <arquivo.jsonl> [--gate]
Sem --gate: relatorio (exit 0). Com --gate: exit 1 se houver violacao.
Pula defs sem properties (sem informacao) e args nao-objeto.
"""
import json
import sys

try:
    import jsonschema
except ImportError:
    print("jsonschema ausente")
    sys.exit(2)

fp = sys.argv[1]
gate = "--gate" in sys.argv
rows = [json.loads(l) for l in open(fp, encoding="utf-8") if l.strip()]
viol = []
checked = 0
for r in rows:
    schemas = {}
    for t in r.get("tools", []):
        fn = t.get("function", {})
        p = fn.get("parameters") or {}
        if isinstance(p, dict) and p.get("properties"):
            schemas[fn.get("name", "")] = p
    for m in r.get("messages", []):
        for tc in (m.get("tool_calls") or []):
            fn = tc.get("function", {})
            sch = schemas.get(fn.get("name", ""))
            if not sch:
                continue
            try:
                a = json.loads(fn.get("arguments", ""))
            except Exception:
                continue
            if not isinstance(a, dict):
                continue
            checked += 1
            try:
                jsonschema.validate(a, sch)
            except jsonschema.ValidationError as e:
                viol.append((r.get("uuid"), fn.get("name"), str(e.message)[:120]))
print(f"calls checadas: {checked} | violacoes: {len(viol)}")
for u, n, w in viol[:20]:
    print(f"  {u} {n}: {w}")
sys.exit(1 if gate and viol else 0)
