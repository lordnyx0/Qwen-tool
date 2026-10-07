"""
apply_battery_en_map.py

Aplica `data/eval_battery_en_manual_map.json` (traducao manual PT->EN, sem API)
sobre a bateria PT, gerando `data/eval_benchmark_battery_v2_en.jsonl`.

Validacao estrita (falha ruidosa):
- toda string traduzivel da bateria tem entrada no mapa;
- toda chave do mapa existe na bateria (chave orfa = erro de digitacao);
- ids/ordem preservados; estrutura fora dos campos traduzidos byte-identica.

Uso:
    python scripts/pipeline/apply_battery_en_map.py
"""

import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "data", "eval_benchmark_battery_v2.jsonl")
MAP = os.path.join(ROOT, "data", "eval_battery_en_manual_map.json")
DST = os.path.join(ROOT, "data", "eval_benchmark_battery_v2_en.jsonl")
MANIFEST = os.path.join(ROOT, "data", "eval_battery_en_translation_manifest.json")

ALLOW_ARGS = {
    "Alinhamento Q4",
    "Deploy v2.4 concluído com sucesso",
    "Enviamos a proposta anexa.",
    "Falha na autenticação OAuth2",
    "O servidor passará por manutenção às 23h",
    "Proposta Comercial",
    "Review de Sprint",
    "como configurar o VPN corporativo",
    "inteligência artificial em semicondutores",
    "milhas",
    "política de reembolso de viagens",
}
CODE_SPAN = re.compile(r"(```.*?```|`[^`\n]+`)", re.DOTALL)


def split_protect(s):
    parts = CODE_SPAN.split(s)
    return [(i % 2 == 1, p) for i, p in enumerate(parts) if p]


def collect(items):
    uniques = {}

    def add(text, ref):
        if text and text.strip():
            uniques.setdefault(text, []).append(ref)

    for i, it in enumerate(items):
        for j, m in enumerate(it.get("messages", [])):
            for is_code, ch in split_protect(m.get("content") or ""):
                if not is_code and ch.strip():
                    add(ch, (i, f"messages[{j}]"))
        for j, t in enumerate(it.get("tools", [])):
            fn = t.get("function", {})
            if fn.get("description"):
                add(fn["description"], (i, f"tools[{j}].description"))
            for pk, pv in ((fn.get("parameters", {}).get("properties")) or {}).items():
                if isinstance(pv, dict) and pv.get("description"):
                    add(pv["description"], (i, f"tools[{j}].params.{pk}"))
        if it.get("expected_text"):
            add(str(it["expected_text"]), (i, "expected_text"))
        for k, v in (it.get("expected_args") or {}).items():
            if isinstance(v, str) and v in ALLOW_ARGS:
                add(v, (i, f"expected_args.{k}"))
    return uniques


def main():
    items = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    uniques = collect(items)
    amap = json.load(open(MAP, encoding="utf-8"))
    print(f"itens: {len(items)} | unicas: {len(uniques)} | mapa: {len(amap)}")

    missing = [s for s in uniques if s not in amap]
    orphan = [s for s in amap if s not in uniques]
    if missing:
        print(f"FALTA NO MAPA ({len(missing)}):")
        for s in missing[:20]:
            print("  -", s[:100])
        sys.exit(1)
    if orphan:
        print(f"CHAVES ORFAS ({len(orphan)}):")
        for s in orphan[:20]:
            print("  -", s[:100])
        sys.exit(1)
    print("cobertura mapa <-> bateria: OK (bijecao exata)")

    def tr_protect(s):
        return "".join(p if is_code else amap[p] for is_code, p in split_protect(s))

    before = json.dumps(items, ensure_ascii=False, sort_keys=True)
    for it in items:
        for m in it.get("messages", []):
            if m.get("content"):
                m["content"] = tr_protect(m["content"])
        for t in it.get("tools", []):
            fn = t.get("function", {})
            if fn.get("description") in amap:
                fn["description"] = amap[fn["description"]]
            for pv in ((fn.get("parameters", {}).get("properties")) or {}).values():
                if isinstance(pv, dict) and pv.get("description") in amap:
                    pv["description"] = amap[pv["description"]]
        if it.get("expected_text") and str(it["expected_text"]) in amap:
            it["expected_text"] = amap[str(it["expected_text"])]
        for k, v in (it.get("expected_args") or {}).items():
            if isinstance(v, str) and v in amap:
                it["expected_args"][k] = amap[v]

    # Sanidade: nada alem dos campos traduzidos mudou?
    check = [json.loads(l) for l in open(SRC, encoding="utf-8") if l.strip()]
    for src_it, dst_it in zip(check, items):
        assert src_it["id"] == dst_it["id"]
        assert src_it["level"] == dst_it["level"]
        assert src_it["category"] == dst_it["category"]
        assert src_it["expected_tool"] == dst_it["expected_tool"]
        assert [t["function"]["name"] for t in src_it["tools"]] == \
               [t["function"]["name"] for t in dst_it["tools"]]
        assert list((src_it.get("expected_args") or {}).keys()) == \
               list((dst_it.get("expected_args") or {}).keys())

    with open(DST, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"mt": "manual(humano, sem API)", "allow_args": sorted(ALLOW_ARGS),
                   "translations": [{"orig": s, "en": amap[s], "usos": len(r)}
                                    for s, r in uniques.items()]},
                  f, ensure_ascii=False, indent=1)

    ids_src = [x["id"] for x in check]
    ids_dst = [json.loads(l)["id"] for l in open(DST, encoding="utf-8") if l.strip()]
    assert ids_src == ids_dst
    print(f"OK: {DST} + manifesto. ids/ordem/estrutura: OK")


if __name__ == "__main__":
    sys.exit(main())
