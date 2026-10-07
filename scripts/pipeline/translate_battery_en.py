"""
translate_battery_en.py

Traduz a bateria holdout PT->EN (modelo treinado em ingles) gerando
`data/eval_benchmark_battery_v2_en.jsonl` + manifesto de auditoria.

Motor: deep_translator (Google, endpoint livre) em BATCH com backoff
(HF exige token nesta rede; opus-mt indisponivel para download).

Regras (conservadoras, auditaveis):
- Traduz: message contents, tool/param descriptions, expected_text,
  e APENAS os valores de expected_args na ALLOWLIST abaixo (prosa PT).
- Preserva byte-identico: ids, ordem, estrutura, tool names, param names,
  required, paths, codigos, enums (lock/unlock/on/off/stop/...), precos,
  tickers, entidades (postgres-prod, ar_condicionado, Curitiba, ...).
- Cercas ``` e spans `code` sao protegidos e nao traduzidos.

Uso:
    python scripts/pipeline/translate_battery_en.py
"""

import json
import os
import re
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "data", "eval_benchmark_battery_v2.jsonl")
DST = os.path.join(ROOT, "data", "eval_benchmark_battery_v2_en.jsonl")
MANIFEST = os.path.join(ROOT, "data", "eval_battery_en_translation_manifest.json")

# Prosa PT em expected_args (levantamento exaustivo dos 144 valores distintos).
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
BATCH = 20
SLEEP = 2.0


def split_protect(s):
    parts = CODE_SPAN.split(s)
    return [(i % 2 == 1, p) for i, p in enumerate(parts) if p]


def batch_translate(texts):
    from deep_translator import GoogleTranslator
    from deep_translator.exceptions import TooManyRequests
    out = {}
    remaining = list(texts)
    attempt = 0
    while remaining:
        attempt += 1
        chunk = remaining[:BATCH]
        try:
            res = GoogleTranslator(source="pt", target="en").translate_batch(chunk)
            for s, t in zip(chunk, res):
                out[s] = (t or s).strip()
            remaining = remaining[BATCH:]
            print(f"  lote ok ({len(out)}/{len(texts)})", flush=True)
            time.sleep(SLEEP)
        except TooManyRequests:
            wait = min(30 * attempt, 300)
            print(f"  429: aguardando {wait}s (tentativa {attempt})", flush=True)
            time.sleep(wait)
        except Exception as e:
            print(f"  erro: {str(e)[:120]} — aguardando 15s", flush=True)
            time.sleep(15)
    return out


def main():
    items = []
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    print(f"itens: {len(items)}", flush=True)

    uniques = {}

    def add(text, ref):
        if text and text.strip():
            uniques.setdefault(text, []).append(ref)

    for i, it in enumerate(items):
        for j, m in enumerate(it.get("messages", [])):
            c = m.get("content") or ""
            for is_code, chunk in split_protect(c):
                if not is_code and chunk.strip():
                    add(chunk, (i, f"messages[{j}]"))
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

    print(f"strings unicas p/ traduzir: {len(uniques)}", flush=True)
    translated = batch_translate(list(uniques.keys()))

    def tr_protect(s):
        return "".join(p if is_code else translated.get(p, p)
                       for is_code, p in split_protect(s))

    for it in items:
        for m in it.get("messages", []):
            if m.get("content"):
                m["content"] = tr_protect(m["content"])
        for t in it.get("tools", []):
            fn = t.get("function", {})
            if fn.get("description") in translated:
                fn["description"] = translated[fn["description"]]
            for pv in ((fn.get("parameters", {}).get("properties")) or {}).values():
                if isinstance(pv, dict) and pv.get("description") in translated:
                    pv["description"] = translated[pv["description"]]
        if it.get("expected_text") and str(it["expected_text"]) in translated:
            it["expected_text"] = translated[str(it["expected_text"])]
        for k, v in (it.get("expected_args") or {}).items():
            if isinstance(v, str) and v in translated:
                it["expected_args"][k] = translated[v]

    manifest = [{"orig": s, "en": translated[s], "usos": len(refs)}
                for s, refs in uniques.items()]

    with open(DST, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump({"mt": "deep_translator/google(pt->en,batch)",
                   "allow_args": sorted(ALLOW_ARGS),
                   "translations": manifest}, f, ensure_ascii=False, indent=1)
    print(f"OK: {DST} ({len(items)} itens) + manifesto ({len(manifest)} traducoes)", flush=True)

    ids_src = [json.loads(l)["id"] for l in open(SRC, encoding="utf-8") if l.strip()]
    ids_dst = [json.loads(l)["id"] for l in open(DST, encoding="utf-8") if l.strip()]
    assert ids_src == ids_dst, "ids/ordem divergiram!"
    print("ids/ordem: OK", flush=True)


if __name__ == "__main__":
    sys.exit(main())
