# data/ — índice

| Pasta / arquivo | Conteúdo | Estado |
|---|---|---|
| `cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl` | **Working file vivo** (N=3211 pós-batch2 2026-10-07; ver `docs/kaggle_sft_handover/14_distribuicao_v39.md` p/ snapshot v39) | corrente |
| `cleaned/curriculum_3000_tool_sft_V1.jsonl` | **Congelado V1** (imutável; ver `docs/kaggle_sft_handover/V1_MANIFEST.md`) | congelado |
| `cleaned/ultradata_toolmind_curated_16k.jsonl` | Matriz primária (16.261 amostras, mineração S5) | fonte |
| `cleaned/curriculum_purified_stats.json` | Stats do working file (regenerado a cada merge) | gerado |
| `cleaned/curriculum_V1_stats.json` | Stats do V1 congelado | congelado |
| `cleaned/` | PURIFIED + V1 + matriz 16k + stats + cadeia `.bak39–51` (backups numerados obrigatórios antes de escrita em massa) | misto |
| `staging/v31/` | Famílias manuais fundidas (`t_err/t_fb/t_ud/t_ca/t_ab3/t_alt/t_alt2`; registro pós-merge, não refundir) | arquivo |
| `staging/v35/`, `v37_*`, `v38_*`, `v39_mt/` | Lotes manuais fundidos (nav, alias, irr, small, beh, mix2, mt; registro pós-merge) | arquivo |
| `staging/quarantine/` | q1/q2/qp5/qp6/qp7/**qp8** (placeholders E39); fora da linhagem de treino | arquivo |
| `staging/dpo_round2/`, `staging/dpo_round4/` | Pares DPO R2 (256+42) e R4 (161 brutos + 55 verificados r4c + fila SFT); DPO enterrado 2026-10-05, somente registro | arquivo |
| `staging/_exemplars/` | Gabaritos manuais de amostra | referência |
| `eval_benchmark_battery_v2.jsonl` | Bateria oficial 121 itens (**fonte PT**; treino é EN — usar `_en` p/ V1) | fonte |
| `eval_benchmark_battery_v2_en.jsonl` | Mesma bateria em EN (tradução manual auditada); eval justo p/ V1 | corrente |
| `eval_battery_en_manual_map.json` | Mapa PT→EN (329 entradas, bijeção validada) + `eval_battery_en_translation_manifest.json` | corrente |
| `eval_probe_canonical_25.jsonl` | Sonda in-distribution 25 itens (builder em `scripts/pipeline/`) | corrente |
| `eval_multiturn_v1.jsonl` | Bateria multi-turn 12 itens EN (correção após observation; overlap 0; runner `scripts/eval/eval_multiturn_battery.py`) | corrente |
| `eval_nav_p5.jsonl` | Bateria navegação P5, 24 itens | corrente |
| `eval_legacy/` | Bateria legada 90 itens (runner `scripts/eval/benchmark_90_battery_q8.py`) | arquivo |
| `eval_results/` | Resultados históricos (somente leitura), por modelo, + `multiturn/`, `p5/`, `sft_v38*`, `sft_dpo_v1/v2/`. Convenção: `battery121*.json`, `probe25.json`, `multiturn*.json` | arquivo |
| `mask_first_turn_68.json` | 67 uuids C3 ativos (mask first-turn; sync `Kaggle/kaggle_sft_data/`) | corrente |
| `split/` | `manifest.jsonl` (3631: executor/calls/tools/token_len/source) + `manifest.html` + `tracks/{narrow,generic}/` (manifest+viewer por track de modelo); views via `scripts/pipeline/make_view.py` | corrente |
| `rl_curriculum/` | Níveis 1-3 de treino RL (**dependência viva** do track GRPO + `sanitize_curriculum.py`; não arquivar) | fonte ativa |
| `rl_rollouts/grpo_frontier_v1.jsonl` | 120 prompts RL frescos (abstenção/python/decoys/multiturn, 0 HARD vs eval; runner `scripts/train/train_grpo_tool_v3.py`) | corrente |
| `rl_rollouts/grpo_pi_v1.jsonl` | 120 rollouts derivados do gate Pi (mesmo schema; run local pi1, early-stop @75) | arquivo |
| `dev_hard_v31_60.jsonl` | Dev-hard 60 itens (15/cat; baseline v3.1 19/60; sync `Kaggle/kaggle_sft_data/`; treino usa via `--dev_eval`) | corrente |
| `tool_grpo_v3_training_log.json` (+`.pre-pi1`) | Telemetria do treino GRPO — **path hardcoded** em `scripts/train/train_grpo_tool_v3.py` e `monitor/Program.cs`; não mover nem renomear | corrente |
| `loop_contribution_analysis.json` | Relatório gerado por `scripts/eval/analyze_loop_contribution.py` | gerado |
| `tool_catalog/` | Catálogo canônico 27 tools (`tools_schema.py` + espelho em `Kaggle/kaggle_sft_data/`) + mapa MCP (`mcp_tools.json/.csv`, via `scripts/pipeline/build_mcp_table.py`) | corrente |

Regra: dataset novo entra por `staging/` → `scripts/pipeline/merge_staging.py` → sync `Kaggle/kaggle_sft_data/`.
Nunca edite `V1` nem `eval_results/`; nunca commite `.bak` para o Kaggle.
Evals ficam na raiz (paths hardcoded nos runners + notebook); resultados seguem a convenção acima.
