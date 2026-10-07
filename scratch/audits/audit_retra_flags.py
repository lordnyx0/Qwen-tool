#!/usr/bin/env python3
"""Audita flags is_retrajectory contra evidencia de erro+recuperacao (somente leitura).

Detector melhorado em relacao ao builder original (que casava so
error/invalid/failed/not found/exception e perdia FileNotFoundError,
sufixos XxxError, exit codes e hifenacoes). NAO altera flags: emite listas para spot-check.
"""
import json
import re

ERR = re.compile(r"(error\w*|err_\w+|\"error\"|fail\w*|FAIL|invalid\w*|invalido|inv.lido|"
                 r"exception|traceback|\w+Error\b|not[\s_\-]*found|no such file|"
                 r"exit (code )?([1-9]\d*)|code [1-9]\d*|returned non-zero|"
                 r"missing|unable|could ?n.?t|didn.?t|refus|denied|timed out|"
                 r"file\w*not\w*found)", re.I)

P = r"C:\Users\Nyx\Downloads\Diversos\Train\Qwen-tool\data\cleaned\curriculum_3000_tool_sft_PURIFIED.jsonl"
rows = [json.loads(l) for l in open(P, encoding="utf-8") if l.strip()]


def evidence(ms):
    for i, m in enumerate(ms):
        if m.get("role") == "tool" and ERR.search(str(m.get("content", ""))):
            if i + 1 < len(ms) and ms[i + 1].get("role") == "assistant":
                return True
    return False


true_no_ev, false_with_ev = [], []
for r in rows:
    ev = evidence(r.get("messages", []))
    if r.get("is_retrajectory") and not ev:
        true_no_ev.append(r["uuid"])
    elif not r.get("is_retrajectory") and ev:
        false_with_ev.append(r["uuid"])
print(f"N={len(rows)} retra_flag={sum(1 for r in rows if r.get('is_retrajectory'))}")
print(f"true_sem_evidencia={len(true_no_ev)} {true_no_ev[:12]}")
print(f"false_com_evidencia={len(false_with_ev)} {false_with_ev[:12]}")
print(f"liquido_se_flipasse={sum(1 for r in rows if r.get('is_retrajectory')) - len(true_no_ev) + len(false_with_ev)}")
