"""Teste unitario do reward_shell_execution + builder (CPU, fixtures inline)."""
import json
import os
import subprocess
import sys

TRAIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "train")
sys.path.insert(0, TRAIN)
from rewards_tool_calling import load_shell_golden, norm_shell_command, reward_command_fidelity, reward_shell_execution

# normalizacao: surface sim, semantica nao
assert norm_shell_command("PowerShell", "Get-Service") == norm_shell_command("powershell", "get-service")
assert norm_shell_command("bash", "ls  /tmp") == norm_shell_command("bash", "ls /tmp")
assert norm_shell_command("powershell", "Get-ChildItem 'C:/Logs'") == \
    norm_shell_command("powershell", 'get-childitem "c:/logs"')
assert norm_shell_command("powershell", "$env:TEMP/x") != norm_shell_command("powershell", "TEMP/x")
assert norm_shell_command("bash", "ls /tmp") != norm_shell_command("bash", "ls /var")

G = {norm_shell_command("powershell", "Get-Service"): {"stdout": "Running  spooler", "exit_code": 0},
     norm_shell_command("bash", "ls /tmp"): {"stdout": "a.txt\nb.txt", "exit_code": 0},
     norm_shell_command("powershell", "Get-ChildItem C:/Logs"): {"stdout": "", "exit_code": 1}}

SAMPLE = {"messages": [{"role": "user", "content": "List services."},
                       {"role": "tool", "name": "powershell", "content": "Running  spooler"},
                       {"role": "assistant", "content": "ok"}]}
EXP = {"name": "powershell", "arguments": {"command": "Get-Service"}}

match = '<tool_call>{"name": "powershell", "arguments": {"command": "Get-Service"}}</tool_call>'
diverge = '<tool_call>{"name": "powershell", "arguments": {"command": "Get-ChildItem C:/Logs"}}</tool_call>'
unknown = '<tool_call>{"name": "powershell", "arguments": {"command": "Get-Process"}}</tool_call>'

r = reward_shell_execution([match, diverge, unknown, "texto puro"],
                           samples=[SAMPLE] * 4, golden=G,
                           expected_call=[EXP] * 4)
assert r == [2.0, 0.5, 0.0, 0.0], r
# sem tabela / sem amostra / abstencao: tudo 0, sem crash
assert reward_shell_execution([match], expected_call=[EXP]) == [0.0]
assert reward_shell_execution([match], samples=[SAMPLE], golden=G,
                              expected_call=[None]) == [0.0]
assert load_shell_golden(None) == {}
assert load_shell_golden(r"C:\nao\existe\tabela.json") == {}
print("reward OK")

# fidelidade: valor exato (normalizado) vs demonstrado
EXP2 = {"name": "powershell", "arguments": {"command": "Get-ChildItem 'C:/Logs'"}}
same_q = '<tool_call>{"name": "powershell", "arguments": {"command": "get-childitem \\"c:/logs\\""}}</tool_call>'
assert reward_command_fidelity([same_q, diverge, unknown, "x"],
                               expected_call=[EXP2] * 4) == [1.0, 0.0, 0.0, 0.0]
assert reward_command_fidelity([same_q], expected_call=[None]) == [0.0]
print("fidelity OK")

# builder com fixtures minimas
TMP = r"C:\Users\Nyx\AppData\Local\Temp\opencode\golden_t"
os.makedirs(TMP, exist_ok=True)
with open(os.path.join(TMP, "ps.jsonl"), "w", encoding="utf-8") as f:
    f.write(json.dumps({"uuid": "a", "idx": 0, "tool": "powershell",
                        "command": "Get-Service", "ok": True, "exit_code": 0,
                        "stdout": "Running", "stderr": "", "refused": ""}) + "\n")
    f.write(json.dumps({"uuid": "b", "idx": 0, "tool": "powershell",
                        "command": "x", "ok": False, "exit_code": 1,
                        "stdout": "", "stderr": "", "refused": "REFUSED"}) + "\n")
with open(os.path.join(TMP, "sh.jsonl"), "w", encoding="utf-8") as f:
    f.write(json.dumps({"uuid": "c", "idx": 0, "tool": "bash",
                        "arguments": {"command": "ls /tmp"}, "ok": True, "exit_code": 0,
                        "stdout": "a", "stderr": "", "refused": ""}) + "\n")
outp = os.path.join(TMP, "tabela.json")
res = subprocess.run([sys.executable, os.path.join(TRAIN, "build_shell_golden.py"),
                      "--ps", os.path.join(TMP, "ps.jsonl"),
                      "--bash", os.path.join(TMP, "sh.jsonl"),
                      "--out", outp], capture_output=True, text=True)
assert res.returncode == 0, res.stderr
t = load_shell_golden(outp)
assert set(t) == {"powershell\x00get-service", "bash\x00ls /tmp"}, set(t)
assert t["powershell\x00get-service"]["stdout"] == "Running"
print("builder OK")
print("ALL PASS")
