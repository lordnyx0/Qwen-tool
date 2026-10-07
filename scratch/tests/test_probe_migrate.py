"""Teste da migracao net.3 -> classifier com o probe real do v38 (CPU)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "scripts", "train"))
import torch
from train_grpo_tool_v3 import LatentHaltingProbe, _migrate_probe_state

FP = r"C:\Users\Nyx\Downloads\Diversos\qwen\checkpoints\qwen3loop_tool_sft_v38\latent_halting_probe.pt"
probe = LatentHaltingProbe(1024)
before = {k: v.clone() for k, v in probe.state_dict().items()}
got = _migrate_probe_state(probe, FP)
assert got == ["net.3.weight->classifier.weight"], got
after = probe.state_dict()
old = torch.load(FP, map_location="cpu", weights_only=True)
assert torch.equal(after["classifier.weight"].cpu().float(), old["net.3.weight"].float())
assert torch.equal(after["shared.0.weight"], before["shared.0.weight"])
# regressor/shared continuam fresh (nao tocados pela migracao)
assert torch.equal(after["regressor.0.weight"], before["regressor.0.weight"])
logits, score = probe(torch.randn(2, 1024))
assert logits.shape == (2, 3) and score.shape == (2,)
# arquivo inexistente -> [] sem crash
assert _migrate_probe_state(probe, r"C:\nao\existe.pt") == []
print("ALL PASS")
