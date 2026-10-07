import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

dim = 1024
vocab = 151936
n_tokens = 3500

lm_head1 = nn.Linear(dim, vocab, bias=False)
lm_head2 = nn.Linear(dim, vocab, bias=False)
lm_head2.weight.data.copy_(lm_head1.weight.data)

h1 = torch.randn(n_tokens, dim, requires_grad=True)
h2 = h1.clone().detach().requires_grad_(True)

target = torch.randint(0, vocab, (n_tokens,))

# Standard full computation
logits1 = lm_head1(h1)
loss1 = F.cross_entropy(logits1, target)
loss1.backward()

# Chunked computation
CHUNK = 1024
total_loss2 = 0.0
for c_start in range(0, n_tokens, CHUNK):
    c_end = min(c_start + CHUNK, n_tokens)
    c_logits = lm_head2(h2[c_start:c_end])
    c_loss = F.cross_entropy(c_logits, target[c_start:c_end], reduction="sum")
    total_loss2 = total_loss2 + c_loss
loss2 = total_loss2 / n_tokens
loss2.backward()

diff_loss = abs(loss1.item() - loss2.item())
diff_grad_w = (lm_head1.weight.grad - lm_head2.weight.grad).abs().max().item()
diff_grad_h = (h1.grad - h2.grad).abs().max().item()

print(f"Loss1 (Full):    {loss1.item():.8f}")
print(f"Loss2 (Chunked): {loss2.item():.8f}")
print(f"Delta Loss:      {diff_loss:.12f}")
print(f"Delta Grad W:    {diff_grad_w:.12f}")
print(f"Delta Grad H:    {diff_grad_h:.12f}")

assert diff_loss < 1e-6, "Loss mismatch!"
assert diff_grad_w < 1e-6, "Grad W mismatch!"
assert diff_grad_h < 1e-6, "Grad H mismatch!"
print("Sanity Check PASSOU com precisao absoluta!")
