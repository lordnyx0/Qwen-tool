"""Teste unitario do scaffold_pool (CPU, sem modelo; pool isolado em /Temp)."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts", "train"))
import scaffold_pool as sp

sp.POOL_DIR = r"C:\Users\Nyx\AppData\Local\Temp\opencode\pool_test"
import shutil
shutil.rmtree(sp.POOL_DIR, ignore_errors=True)

sp.ensure_files(["frontier_python"])
assert os.path.isfile(sp.pool_path("frontier_python"))
assert "print" in sp.get_best("frontier_python").lower()
p = sp.build_refine_prompt("Compute 2+2", "frontier_python")
assert "2+2" in p and "<tool_call>" not in p

best, mean = sp.update("frontier_python", "plano A", 6.0)
assert best and mean == 6.0
best, mean = sp.update("frontier_python", "plano A", 4.0)
assert not best and abs(mean - 5.0) < 1e-9
best, _ = sp.update("frontier_python", "plano B", 1.0)
assert not best
assert sp.get_best("frontier_python") == "plano A"
# ensure_files nao sobrescreve vencedor
sp.ensure_files(["frontier_python"])
assert sp.get_best("frontier_python") == "plano A"
# nivel desconhecido: seed generica, sem crash, update funciona
assert sp.get_best("inexistente") != ""
sp.ensure_files(["ws_powershell"])
assert "Identify the request" in sp.get_best("ws_powershell")
best, mean = sp.update("ws_powershell", "plano W", 3.0)
assert best and mean == 3.0
shutil.rmtree(sp.POOL_DIR, ignore_errors=True)
print("ALL PASS")
