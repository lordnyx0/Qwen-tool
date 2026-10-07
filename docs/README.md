# Documentação Técnica: Ecossistema Qwen3Loop-0.6B

Bem-vindo à base de documentação oficial do projeto **Qwen3Loop-0.6B**, um modelo compacto de raciocínio recorrente com 594M parâmetros físicos e unrolling para 56 camadas lógicas em formato GGUF.

---

## Índice de Documentos Técnicos

Abaixo está o índice completo dos guias de arquitetura, dados, treinamento e quantização:

| Documento | Descrição |
| :--- | :--- |
| 📘 [`metodologia_treino_base_sft.md`](metodologia_treino_base_sft.md) | **Método do Primeiro Treino (Base SFT):** Formulação da Multi-Exit Loss ($w_1=0.4, w_2=1.0$), cabeça Latent Halting Probe, otimizador 8-bit AdamW e curadoria com Laya. |
| ⚙️ [`arquitetura_recorrente.md`](arquitetura_recorrente.md) | **Arquitetura de Recorrência em Profundidade:** Estrutura física de 28 camadas vs. 56 lógicas, indexação virtual de KV Cache, Dynamic Cache e sonda latente de complexidade. |
| 📊 [`curriculo_e_datasets.md`](curriculo_e_datasets.md) | **Currículo de Tool Calling:** Estrutura em 3 níveis (N1 Fundação/Abstenção, N2 Discriminação de Schemas, N3 Matemática com `execute_python`) e ChatML template. |
| 🚀 [`treinamento_tool_calling_sft.md`](treinamento_tool_calling_sft.md) | **Guia do Treino Supervisionado (SFT):** Pipeline Full Fine-Tuning dos 594M parâmetros, máscara de perda sobre o assistente, configurações e execução na nuvem Kaggle. |
| 📦 [`unrolling_e_quantizacao_gguf.md`](unrolling_e_quantizacao_gguf.md) | **Unrolling e Quantização GGUF:** Desdobramento de 28 para 56 camadas compatíveis com `llama.cpp`, quantização em Q8_0 (1.11 GB) e Q4_K_M (0.68 GB). |
| 🎯 [`benchmark_e_avaliacao.md`](benchmark_e_avaliacao.md) | **Bateria de Avaliação (90 Casos Holdout):** Protocolo cego de testes, métricas de conformidade JSON, correspondência de argumentos e execução sandbox de Python. |
| 🔬 [`analise_ric_grpo_e_quantizacao.md`](analise_ric_grpo_e_quantizacao.md) | **Estudo Post-Mortem de RL e LoRA:** Demonstração analítica da aniquilação de matrizes LoRA no arredondamento Q8_0 e análise de gradientes zero no GRPO. |
| 📁 [**`kaggle_sft_handover/`**](kaggle_sft_handover/README.md) | **Dossiê de Handover SFT (Kaggle Dual T4):** Diagnóstico post-mortem do dataset legado, especificação canônica do novo dataset, infraestrutura Kaggle e checklist passo a passo. |

---

## Fluxo Rápido de Trabalho

```
1. Dados e Currículo ──> 2. Full-FT SFT ──> 3. Checkpoint Unrolling ──> 4. Quantização Q8_0 ──> 5. Holdout Benchmark
   (docs/curriculo)        (docs/treino)        (docs/unrolling)            (llama-quantize)        (docs/benchmark)
```
