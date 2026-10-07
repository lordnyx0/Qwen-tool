# scripts/ — índice

Executar sempre da raiz do repo (`python scripts/<pasta>/<arq>.py`).

| Pasta | Conteúdo | Status |
|---|---|---|
| `pipeline/` | Curadoria ativa V1/V2: `fix_curriculum_3000.py`, `build_purified_sft_dataset.py`, `merge_staging.py`, `validate_sft.py` (gates do dataset), `validate_docs.py` (gates doc↔código), `build_probe_canonical_25.py`, `build_holdout_battery_v2.py`, `translate_battery_en.py` (via API, superseded), `apply_battery_en_map.py` (tradução manual vigente), `sanitize_and_curate_tool_use.py`, `sanitize_curriculum.py`, `check_eval_overlap.py`, `check_v31_diversity.py` (gate anti-template V3.1) | **corrente** |
| `train/` | Treino SFT/GRPO + rewards (`train_sft_tool_qwen3loop.py` é o SFT; cópia de deploy vive em `Kaggle/kaggle_sft_data/`; `train_grpo_tool_v3.py` é o GRPO fronteira — init V3, KL ancorado, `reward_length`, sampler em `data/rl_rollouts/`) | corrente |
| `eval/` | Benchmarks e diagnósticos (`benchmark_battery_v2.py` é o oficial single-turn; `eval_multiturn_battery.py` é o multi-turn multi-modelo; `benchmark_90_battery_q8.py` lê `data/eval_legacy/`) | corrente |
| `quantize/` | Unroll + GGUF pós-treino (`build_f16_base_from_safetensors.py` gera o template F16 ausente; `build_and_quantize_tool_sft_gguf.py` faz inject→unroll→Q8) | corrente (pós-treino) |
| `deploy/` | Empacotamento/upload Kaggle | corrente |
| `build_archive/` | Builders superseded (8 gerações antigas, só referência) | arquivo |
| `legacy_sft_v1/` | Pipeline legado v1 isolado | arquivo |

Nota: arquivos movidos em 2026-09-27 tiveram `ROOT = dirname(__file__)/..` ajustado para `/../..`.
Links `file:///.../scripts/*.py` nos docs antigos (06) podem apontar o caminho flat — este README é o índice canônico.
