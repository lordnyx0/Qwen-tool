"""Teste do backend plumb com urllib mockado (CPU, sem rede/modelo)."""
import io
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts", "train"))
from rewards_tool_calling import JUDGE_CRITERION, judge_veto, make_plumb_judge


class FakeResp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def fake_urlopen_factory(p_true):
    def _fake(req, timeout=None):
        body = json.loads(req.data.decode())
        assert req.full_url.endswith("/v1/systemone"), req.full_url
        assert body["questions"]["veto"]["type"] == "noul"
        assert "gamed" in body["questions"]["veto"]["instructions"]
        return FakeResp({"answers": {"veto": {"noul": p_true}}})
    return _fake


real = urllib.request.urlopen
urllib.request.urlopen = fake_urlopen_factory(0.93)
assert make_plumb_judge()("caso X") == 1
urllib.request.urlopen = fake_urlopen_factory(0.2)
assert make_plumb_judge()("caso X") == 0
assert make_plumb_judge(threshold=0.1)("caso X") == 1
urllib.request.urlopen = real
print("backend OK; criterio:", JUDGE_CRITERION[:60] + "...")

# fim-a-fim no veto com consulta realista (mockada)
urllib.request.urlopen = fake_urlopen_factory(0.95)
fn = make_plumb_judge()
gaming = "The PowerShell operation has been executed successfully."
EXP = {"name": "powershell", "arguments": {"command": "Get-Service"}}
assert judge_veto([gaming], expected_call=[EXP], tasks=["t"], judge_fn=fn) == [True]
urllib.request.urlopen = real
print("ALL PASS")
