"""build_shell_golden.py — consolida artefatos de coleta real em tabela de fixture.

Uso: python scripts/train/build_shell_golden.py --ps <ps_results.jsonl>
       --bash <bash_results.jsonl> [--s6 <s6_ps_results.jsonl> ...] --out <tabela.json>
Formato: {"<tool>\\x00<command>": {"stdout": str, "exit_code": int}}.
So entram linhas ok (exit 0) ou falha genuina documentada; refused/timeout ficam
de fora (sem sinal). Tabelas reais na etapa 2.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewards_tool_calling import norm_shell_command

TOOLS = {"powershell", "bash"}


def _text(v):
    if isinstance(v, dict):
        v = v.get("value", "")
    return v if isinstance(v, str) else str(v or "")


def _exit(v):
    if isinstance(v, dict) and "value" in v:
        return int(v["value"])
    return v if isinstance(v, int) else -1


def load_results(fp, default_tool=None):
    rows = []
    with open(fp, encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", action="append", default=[])
    ap.add_argument("--bash", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    table, skipped = {}, {"refused": 0, "timeout": 0, "null": 0}
    for fp in a.ps + a.bash:
        tool_default = None
        for r in load_results(fp):
            if r.get("refused"):
                skipped["refused"] += 1
                continue
            code = _exit(r.get("exit_code"))
            if code in (124, -1):
                skipped["timeout" if code == 124 else "null"] += 1
                continue
            tool = r.get("tool", tool_default or "")
            args = r.get("arguments", {}) or {}
            if tool == "bash":
                cmd = args.get("command", "")
            elif tool == "powershell":
                cmd = (args.get("command", "")
                       if isinstance(args, dict) else "")
                if not cmd and isinstance(r.get("command"), str):
                    cmd = r["command"]
            else:
                continue
            if tool not in TOOLS or not cmd:
                continue
            table[norm_shell_command(tool, cmd)] = {"stdout": _text(r.get("stdout")),
                                                    "exit_code": code}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False)
    print(f"entradas={len(table)} puladas={skipped} -> {a.out}")


if __name__ == "__main__":
    main()
