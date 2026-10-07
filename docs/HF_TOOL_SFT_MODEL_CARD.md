---
language:
- en
license: apache-2.0
tags:
- tool-calling
- function-calling
- qwen3loop
- recursive-reasoning
- loop-architecture
- gguf
- unrolled
- agentic-sft
pipeline_tag: text-generation
---

# Qwen3Loop-0.6B-Tool-Calling-SFT (v1)

**Qwen3Loop-0.6B-Tool-Calling-SFT** is a narrow tool-calling specialist: the 0.6B-parameter
recurrent Qwen3Loop architecture, fully fine-tuned on a curated curriculum of **1,983 strict
tool-calling samples** (single-turn `user → assistant(tool_calls)`, mid-trajectory corrections,
terminal/memory/API calls, Qwen Hermes `<tool_call>` format).

### Core architecture
- **Physical parameters:** 28 transformer blocks (`hidden_size: 1024`, `intermediate_size: 3072`, `heads: 16`).
- **LoopSplit:** prefix layers 0..6 (once) + middle stack 7..20 (**3 recurrent passes** = 42 layer passes) + suffix 21..27 (once) = **56 logical layers** ($7 + 14 \times 3 + 7$).
- **Training:** full fine-tuning (not LoRA), 2×T4 DDP, 1 epoch on the frozen V1 curriculum. Lineage: Qwen3Loop-0.6B SFT family (`Lordnyx/qwen3loop-0.6b-sft-deep-supervision-v1`).

---

## Measured capabilities (Q8_0, RTX 3060, `temperature=0.0`)

121-item English holdout battery (50 tools, 8 domains — translated from the original PT battery so the model is tested in its training language):

| Slice | Score | Reading |
| :--- | :---: | :--- |
| **Single-tool precise (L1)** | **30/30 (100%)** | Known tool + arguments: solved |
| **Multi-tool discrimination (L2)** | **35/41 (85.4%)** | Strong; fails on near-neighbor finance decoys |
| **Python reasoning (L3)** | **10/30 (33.3%)** | Weak trigger (see limitations) |
| **Negative abstention (L4)** | **10/20 (50.0%)** | Unreliable (see limitations) |
| **Overall strict accuracy** | **85/121 (70.2%)** | Format adherence 83.5%, tool selection 81.8% |

12-item multi-turn correction battery (fixed error observation → model must correct; identical prompts, GPU):

| Model | Score |
| :--- | :---: |
| **This checkpoint (tool-SFT)** | **8/12 (66.7%)** — only model that retries identically on timeout and broadens failed queries |
| Stock Qwen3-0.6B Q8_0 | 5/12 (41.7%) |
| Pre-tool SFT base (same lineage) | 3/12 (25.0%) — base SFT had nearly suppressed tool use; tool-SFT restored it (+42pp over its own base) |

---

## Explicit limitations — read before deploying as an agent

1. **English only.** ~99% of training is English; non-English prompts measurably degrade tool selection and triggering (same battery in PT scored ~15pp lower, partial run).
2. **Does not reliably abstain.** It calls tools where it should stay silent in ~50% of negative cases (5/5 probe abstention items failed). Gate it externally if calls have cost or side effects.
3. **Misses the code trigger.** On ~1/3 of direct quantitative prompts it answers in prose instead of calling `execute_python`. It was trained on file-based heredoc Python, not bare `print()` snippets.
4. **Unseen tools confuse it.** Finance tools (`get_stock_quote` vs `get_company_financials`) and tools outside the 15-tool training catalog cause wrong-tool or no-call failures. In-distribution invocation is 100%; transfer is not.
5. **Validated single-turn only.** Multi-turn iterative behavior is measured on 12 items only; long agentic loops are untested.
6. **0.6B capacity ceiling.** Obscure Unix flags, fine-grained decoys, and invented entity names (e.g. hallucinated cmdlet variants) reflect capacity limits, not just data.
7. **Stylistic biases.** Verbose `<think>` preambles and empty-query memory searches mirror training priors; harmless but noisy in logs.

This is a **reliable invoker of known tools in single turns**, not yet a fully autonomous tool specialist. V2 targets abstention, code triggering, and finance-decoy discrimination.

---

## Dynamic depth: entropy-gated loop skipping

[`latent_halting_probe.pt`](latent_halting_probe.pt) is an ultralight MLP head trained on intermediate
prompt **prefill states**. At inference it estimates per-request entropy/complexity and **skips the
recurrent middle-stack passes when they are not needed**, falling back to a shallower pass —
higher throughput on easy requests, full 56-layer depth when it matters. Wire it through the
forward hooks in [`engine/qwen3loop_python/`](engine/qwen3loop_python/).
(Note: the 91.5% routing accuracy quoted for the base-SFT lineage probe was not re-measured on this
checkpoint; treat the gate as a speed heuristic and validate on your workload.)

---

## Model files

### 1. Universal unrolled models (stock llama.cpp / Ollama / LM Studio, arch `qwen3`, 56 blocks)
* [`unrolled_modelo_qwen3loop_tool_sft_q8_0.gguf`](unrolled_modelo_qwen3loop_tool_sft_q8_0.gguf) (**1.03 GB**): Q8_0. Recommended for local engines.
* [`unrolled_modelo_qwen3loop_tool_sft_f16.gguf`](unrolled_modelo_qwen3loop_tool_sft_f16.gguf) (**1.94 GB**): full-precision FP16.

### 2. Compact native looped models (28 physical blocks; need a loop-aware engine or the patch below)
* [`modelo_qwen3loop_tool_sft_q8_0.gguf`](modelo_qwen3loop_tool_sft_q8_0.gguf) (**0.64 GB**)
* [`modelo_qwen3loop_tool_sft_f16.gguf`](modelo_qwen3loop_tool_sft_f16.gguf) (**1.12 GB**)

### 3. PyTorch weights, probe & engine
* [`model.safetensors`](model.safetensors) (**1.11 GB**): full fine-tuned weights.
* [`latent_halting_probe.pt`](latent_halting_probe.pt): entropy halting MLP (see above).
* [`config.json`](config.json), [`tokenizer.json`](tokenizer.json), [`tokenizer_config.json`](tokenizer_config.json), [`generation_config.json`](generation_config.json), [`chat_template.jinja`](chat_template.jinja).
* [`engine/qwen3loop_python/`](engine/qwen3loop_python/): modeling + config + minimal forward (`modeling_qwen3loop.py`, `configuration_qwen3loop.py`, `modeling_qwen3loop_minimal.py`).
* [`engine/llamacpp_patch/`](engine/llamacpp_patch/): llama.cpp loader patch reference for the looped architecture.

---

## Usage

PyTorch (custom modeling ships in-repo, no `trust_remote_code` needed):

```python
import sys
sys.path.insert(0, "engine")
from transformers import AutoTokenizer
from qwen3loop_python.modeling_qwen3loop import Qwen3LoopForCausalLM
import torch

tok = AutoTokenizer.from_pretrained("Lordnyx/qwen3loop-0.6b-tool-calling-sft-v1")
model = Qwen3LoopForCausalLM.from_pretrained(
    "Lordnyx/qwen3loop-0.6b-tool-calling-sft-v1", torch_dtype=torch.bfloat16).cuda()
prompt = tok.apply_chat_template(
    [{"role": "user", "content": "Convert 42 km to miles."}],
    tools=[{"type": "function", "function": {
        "name": "convert_measurement_units",
        "description": "Converts physical quantities.",
        "parameters": {"type": "object", "properties": {
            "value": {"type": "number"}, "from_unit": {"type": "string"},
            "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}}],
    tokenize=False, add_generation_prompt=True)
```

GGUF: load any `unrolled_*` file in Ollama / LM Studio / llama.cpp (`-ngl 99` for full GPU offload).

```json
{ "temperature": 0.0, "top_p": 1.0, "repeat_penalty": 1.0, "context_length": 8192 }
```
*(Use `temperature: 0.0` for deterministic tool calls; the model was evaluated greedy.)*
