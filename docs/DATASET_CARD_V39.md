---
license: unknown
language:
- en
tags:
- tool-calling
- sft
- agents
---

# Qwen3Loop Tool SFT — training dataset v39 (private working snapshot)

Curriculum for the Qwen3Loop-0.6B tool-calling specialist (Pi coding agent).
N=3567, single JSONL (`curriculum_3000_tool_sft_PURIFIED.jsonl`), one sample
per line: `uuid`, `source`, `tools[]` (OpenAPI-ish defs), `messages[]`
(Hermes `tool_calls` with `id`, plus `tool` observations).

## Project

Small (0.6B) English-only specialist trained by full-parameter SFT from a
fixed base, 1 dataset ↔ 1 model per version, frozen evals, gates before every
merge. Current best model: tool-calling-sft-v38 (epoch 2) — battery 103/121,
5/5 ship bars, live Pi gate 2.5/6. A surgical follow-up (140 samples) did not
move gate behavior and was NOT shipped.

## Composition by source

| source | n | what |
|---|---|---|
| purified_canonical | 1398 | ToolMind/legacy multi-tool flows, curated |
| manual_v37_small | 640 | small-catalog copies (1–4 tools, Pi-like band) |
| manual_s5 | 271 | scripted tool flows |
| manual_v35 | 200 | P4 navigation, real executed outputs |
| manual_v37_alias | 200 | P4 alias copies (30 tool-name variants) |
| manual_v31 | 172 | workspace calibration (incl. clarifications) |
| manual_v37 | 156 | irrelevance-with-catalog refusals |
| manual_v38_beh | 140 | surgical behavior pack (run-code, no-tool, policy up-samples, replay) |
| manual_t_abs/trig/trig2/dec/dec2/sem/cmd/cal | 330 | abstention, triggers, decisions, calibration |
| manual_v34 | 50 | T-ALT navigation |

Catalog sizes 1–20 (median ~9–12; 25.9% small 1–4). 1229 unique called tools.
Masks used at train time: repeat-after-error (C1), schema-violation (C1b),
error-followed (C1c), false-first-turn list (C3) — see train script.

## Gates (all green at snapshot)

13 validation rules (JSON, args, declared-calls, descriptions-only-declared,
no-dup-names, required⊆properties…), eval-overlap 0 HARD (167 signatures),
JSON-schema call validation 0 violations, numbered `.bakN` backups.

## Known limitations (for the next developer)

- Probe hard-abstention 0/5; multiturn mass thinned by collapsing (94% 1-turn).
- 53.5% of called tools appear once (generalization, thin mastery).
- ~49% of defs lack `properties` (denominator inflated by distractor copies).
- Live-gate bugs (open): code-echo instead of execute, invented paths,
  identical retries, over-broad listing. See gate logs; surgical SFT did not fix.
- 2/3567 samples exceed 5000 tokens (silently skipped at train).
