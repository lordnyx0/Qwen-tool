"""Teste unitario do veto Ornith-L2 com juiz mock (CPU, sem modelo/servidor)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts", "train"))
from rewards_tool_calling import judge_veto

EXP = {"name": "powershell", "arguments": {"command": "Get-Service"}}
genuine = '<tool_call>{"name": "powershell", "arguments": {"command": "Get-Service"}}</tool_call> done.'
gaming = 'The PowerShell operation has been executed successfully.'
plain_fail = 'I could not find the service.'
calls_made = []

none_fn = None
mock_yes = lambda p: (calls_made.append(p), "1")[1]
mock_no = lambda p: "0"


def boom(p):
    raise RuntimeError("juiz fora do ar")


# flag desligada: nunca veta
assert judge_veto([gaming], expected_call=[EXP], tasks=["t"], judge_fn=none_fn) == [False]
# genuina com call: nem consulta o juiz
calls_made.clear()
assert judge_veto([genuine], expected_call=[EXP], tasks=["t"], judge_fn=mock_yes) == [False]
assert calls_made == [], "juiz nao deve ser consultado sem suspeita"
# suspeita + juiz 1: veta
assert judge_veto([gaming], expected_call=[EXP], tasks=["t"], judge_fn=mock_yes) == [True]
assert len(calls_made) == 1 and "GAMED" in calls_made[0]
# suspeita + juiz 0: libera
assert judge_veto([gaming], expected_call=[EXP], tasks=["t"], judge_fn=mock_no) == [False]
# abstencao: fora do veto mesmo com texto de sucesso
assert judge_veto([gaming], expected_call=[None], tasks=["t"], judge_fn=mock_yes) == [False]
# falha simples sem claim: sem suspeita
assert judge_veto([plain_fail], expected_call=[EXP], tasks=["t"], judge_fn=mock_yes) == [False]
# juiz quebrou: fail-open (nao mata o treino por flap de infra)
assert judge_veto([gaming], expected_call=[EXP], tasks=["t"], judge_fn=boom) == [False]
print("ALL PASS")
