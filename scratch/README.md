# scratch/ — índice (experimentos e utilitários; nada aqui entra no treino)

Rodar sempre da raiz (`python scratch/<pasta>/<arq>.py`). Caminhos relativos a `data/` continuam válidos.

| Pasta | Conteúdo |
|---|---|
| `audits/` | Auditorias e inspeções de dataset (22): `audit_*`, `scan_*`, `check_dataset_*`, `inspect_*`, `measure_token_distributions`, `test_hermes_*`. Inclui `audit_curriculum_3000.py` (auditoria amostral do legado, ref. 06 §A). |
| `deploy/` | Deploy Kaggle e notebooks (push dataset/kernel, rebuild, filtros, patch de notebook). Inclui `push_curriculum_3000.py` (dataset+kernel). |
| `monitoring/` | Monitores de treino remoto (`monitor_v13`-`v18`, `poll_v12.py`, `stream_sft_live.py`). Monitor atual: `kaggle kernels status` + API (nota: o `poll_logs` citado nos docs **não existe** — usar `monitoring/` ou CLI). |
| `tests/` | Testes unitários/integração e experimentos (equivalência de checkpoint/loss, SDPA, gerações L1/L3, abstention, VRAM). |
| `infra/` | Utilitários pontuais: `organize_workspace.py`, `execute_purge_tier1_to_4.py` (**destrutivo**, só com backup), `HF_DATASET_README.md`. |
| `kaggle_v11_output/` | Saídas históricas de kernel v11 (somente leitura). |

Regra: rascunho novo entra na pasta temática; nada solto na raiz do `scratch/`.
