#!/usr/bin/env python3
"""Detecta amostras duplicadas em um dataset ChatML de Tool SFT (.jsonl).

Somente leitura por padrão. Nunca altera o arquivo de entrada.

Assinatura de duplicata (padrão):
    primeiro user normalizado + sequência de (tool_name, arguments canônicos)
Modo --user-only: considera só o texto do primeiro user (duplicata "fraca").

Uso:
    python scratch/audits/check_duplicates.py data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl
    python scratch/audits/check_duplicates.py data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl --manifest data/split/manifest.jsonl --top 15
    python scratch/audits/check_duplicates.py data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl --user-only
    python scratch/audits/check_duplicates.py data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl --write-clean <saida>.jsonl
    python scratch/audits/check_duplicates.py data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl --report dup_report.md

Saída --write-clean: vai para staging ou /Temp, nunca sobrescreve a entrada.

Código de saída: 0 = sem duplicatas, 1 = duplicatas encontradas, 2 = erro.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict


def norm_text(s):
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def canon_args(a):
    """arguments vem como string JSON; canonicaliza para ignorar ordem de chaves/espacos."""
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            return a.strip()
    return json.dumps(a, sort_keys=True, ensure_ascii=False)


def first_user(msgs):
    for m in msgs:
        if m.get("role") == "user":
            return norm_text(m.get("content"))
    return ""


def tool_sequence(msgs):
    seq = []
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            seq.append((fn.get("name", ""), canon_args(fn.get("arguments"))))
    return tuple(seq)


def signature(row, user_only):
    msgs = row.get("messages", [])
    u = first_user(msgs)
    return u if user_only else (u, tool_sequence(msgs))


def load_jsonl(path):
    rows, errors = [], 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                errors += 1
                print(f"[aviso] linha {i} invalida, ignorada", file=sys.stderr)
    return rows, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset")
    ap.add_argument("--manifest", help="data/split/manifest.jsonl (opcional; adiciona executor/source)")
    ap.add_argument("--user-only", action="store_true", help="duplicata so pelo texto do user")
    ap.add_argument("--prefer-retra", action="store_true",
                    help="no dedup, mantem a retrajetoria do grupo (senao, 1a ocorrencia)")
    ap.add_argument("--top", type=int, default=10, help="quantos grupos exibir")
    ap.add_argument("--report", help="salva o relatorio Markdown neste caminho")
    ap.add_argument("--write-clean", help="grava copia deduplicada (mantem 1 por grupo)")
    args = ap.parse_args()

    if args.write_clean and args.write_clean == args.dataset:
        print("ERRO: --write-clean nao pode sobrescrever o arquivo de entrada.", file=sys.stderr)
        return 2

    rows, bad = load_jsonl(args.dataset)
    meta = {}
    if args.manifest:
        with open(args.manifest, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    meta[r["uuid"]] = r

    groups = defaultdict(list)
    for r in rows:
        groups[signature(r, args.user_only)].append(r)

    if args.prefer_retra:
        for g in groups.values():
            g.sort(key=lambda r: (not r.get("is_retrajectory"), r.get("uuid", "")))

    dups = [g for g in groups.values() if len(g) > 1]
    rows_in_dups = sum(len(g) for g in dups)
    removable = rows_in_dups - len(dups)
    unique = len(rows) - removable

    def tag(r, k):
        return meta.get(r.get("uuid", ""), {}).get(k) or r.get("source" if k == "source" else k, "?")

    # cruzamento entre fontes: duplicata entre fontes diferentes e o pior caso p/ vazamento
    cross = sum(1 for g in dups if len({tag(r, "source") for r in g}) > 1)
    src_counter = Counter(tag(r, "source") for g in dups for r in g[1:])  # copias removiveis
    exe_counter = Counter(tag(r, "executor") for g in dups for r in g[1:]) if meta else None
    sizes = Counter(len(g) for g in dups)

    modo = "so user" if args.user_only else "user + tool calls"
    out = []
    out.append(f"# Relatorio de duplicatas ({modo})\n")
    out.append(f"- Amostras lidas: {len(rows)} (linhas invalidas: {bad})")
    out.append(f"- Grupos duplicados: {len(dups)}")
    out.append(f"- Linhas envolvidas em duplicatas: {rows_in_dups} ({rows_in_dups / max(len(rows), 1):.1%})")
    out.append(f"- Copias removiveis (mantendo 1 por grupo): {removable}")
    out.append(f"- Amostras unicas apos dedup: {unique}")
    out.append(f"- Grupos que cruzam fontes diferentes: {cross}")
    out.append(f"- Tamanho dos grupos: {dict(sorted(sizes.items()))}")
    out.append("\n## Copias removiveis por source\n")
    out += [f"- {s}: {n}" for s, n in src_counter.most_common()]
    if exe_counter:
        out.append("\n## Copias removiveis por executor\n")
        out += [f"- {s}: {n}" for s, n in exe_counter.most_common()]
    out.append(f"\n## Top {args.top} grupos (maiores primeiro)\n")
    for g in sorted(dups, key=len, reverse=True)[: args.top]:
        user = first_user(g[0]["messages"])[:90]
        out.append(f"- x{len(g)} \"{user}\": " + ", ".join(r.get("uuid", "?") for r in g))

    text = "\n".join(out)
    print(text)
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")

    if args.write_clean:
        seen = set()
        kept = 0
        with open(args.write_clean, "w", encoding="utf-8") as f:
            for r in rows:
                s = signature(r, args.user_only)
                if s in seen:
                    continue
                seen.add(s)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
        print(f"\n[ok] copia deduplicada: {kept} amostras -> {args.write_clean}")

    return 1 if dups else 0


if __name__ == "__main__":
    sys.exit(main())
