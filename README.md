# Qwen3Loop-0.6B: Compact Depth-Recurrent Tool Agent

Repositório oficial de pesquisa, engenharia de modelo e fine-tuning do **Qwen3Loop-0.6B**, um modelo de linguagem de 594M parâmetros físicos com arquitetura recorrente em profundidade (*Depth-Recurrent Transformer*), unrolling para 56 camadas lógicas e especialização nativa em **Tool Calling** e execução de código Python (`execute_python`).

---

## 📁 Estrutura do Repositório

```
qwen/
├── AGENTS.md                    # Convenções obrigatórias p/ agentes (leia primeiro)
├── data/                        # Datasets e benchmarks (ver data/README.md)
│   ├── cleaned/                 # PURIFIED (vivo), V1 (congelado), ultradata 16k, _archive/
│   ├── staging/s5/              # Lotes S5 validados aguardando merge
│   ├── eval_benchmark_battery_v2.jsonl  # Bateria oficial 121 itens
│   ├── eval_probe_canonical_25.jsonl    # Sonda in-distribution 25 itens
│   ├── eval_legacy/             # Bateria legada 90 itens
│   ├── eval_results/            # Resultados históricos (somente leitura)
│   ├── rl_curriculum/           # Níveis 1-3 de treino RL (arquivo)
│   └── tool_catalog/            # Catálogo canônico 15 tools
├── docs/                        # Documentação (ver docs/kaggle_sft_handover/README.md + CHANGELOG.md)
├── engine/                      # Motor customizado de recorrência (fonte canônica)
│   └── qwen3loop_python/        # Modelagem HF (modeling_qwen3loop, config)
├── checkpoints/                 # Checkpoints (sft_final, tool_sft_final, grpo_*)
├── models/gguf/                 # Modelos exportados em GGUF
├── Kaggle/                       # Tudo de deploy/execução Kaggle
│   ├── kaggle_sft_data/             # Pacote de deploy SFT (espelho do PURIFIED + engine + train)
│   ├── kaggle_kernel/               # Notebook + metadata de execução no Kaggle
│   ├── kaggle_kernel_output/        # Saídas de kernel (arquivo)
│   └── _legacy/                     # Bundles GRPO antigos (arquivo)
├── pack_agentic/                # Export do loop SFT + base (outro ambiente)
│   ├── kaggle_bundle/               # Bundle GRPO legado (outro track)
├── logs/                        # Telemetria e execuções (kaggle_runs/, grpo_run_history/)
├── scratch/                     # Experimentos (ver scratch/README.md; nada entra no treino)
├── scripts/                     # pipeline/ train/ eval/ quantize/ deploy/ build_archive/ (ver scripts/README.md)
├── .gitignore / requirements.txt
└── .codegraph/ unsloth_compiled_cache/  # caches locais, ignorar
```

---

## 🚀 Guia Rápido

### 1. Documentação Técnica Completa
Consulte a pasta [`docs/`](docs/README.md) para todos os detalhes teóricos, matemáticos e empíricos da arquitetura e dos treinamentos.

### 2. Avaliação no Modelo GGUF Q8_0 (bateria oficial 121 + sonda 25)
```bash
python scripts/eval/benchmark_battery_v2.py \
    --model_path models/gguf/legacy_v1/unrolled_modelo_qwen3loop_tool_sft_q8_0.gguf \
    --tokenizer_dir models/gguf/legacy_v1 \
    --battery_path data/eval_benchmark_battery_v2_en.jsonl
python scripts/eval/benchmark_battery_v2.py \
    --model_path models/gguf/legacy_v1/unrolled_modelo_qwen3loop_tool_sft_q8_0.gguf \
    --tokenizer_dir models/gguf/legacy_v1 \
    --battery_path data/eval_probe_canonical_25.jsonl
```

### 3. Treinamento Supervisionado de Tool Calling (Kaggle)
Dataset V1 congelado (1983 amostras, ver `docs/kaggle_sft_handover/V1_MANIFEST.md`). Kernel [`lordnyx0/qwen3loop-tool-sft`](https://www.kaggle.com/code/lordnyx0/qwen3loop-tool-sft) v21 disparado em 2026-09-27. Para verificar o status via terminal local:
```powershell
$env:KAGGLE_API_TOKEN = (kaggle auth print-access-token).Trim()
kaggle kernels status lordnyx0/qwen3loop-tool-sft
```
