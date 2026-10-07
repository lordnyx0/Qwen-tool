# 06. Repositórios de Origem dos Datasets e Catálogo de Scripts

Este documento consolida todos os links externos de repositórios no Hugging Face, caminhos de datasets locais, IDs de datasets no Kaggle e o inventário completo de scripts utilizados no pipeline de Tool Calling do **Qwen3Loop-0.6B**.

---

## 🌐 1. Repositórios de Datasets no Hugging Face

Abaixo estão os links diretos para cada uma das fontes públicas exploradas e utilizadas:

| Dataset / Repositório | Link Oficial no Hugging Face | Arquivo / Split Utilizado | Descrição Técnica |
| :--- | :--- | :--- | :--- |
| **NVIDIA Nemotron Terminal Corpus** | [huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus](https://huggingface.co/datasets/nvidia/Nemotron-Terminal-Corpus) | `dataset_adapters/code.parquet`<br>`dataset_adapters/swe.parquet` | **366.154 trajetórias** de terminal Linux nativo (`bash`), resolução de problemas de engenharia e SWE-bench no formato Terminus-2. |
| **Hermes Agent Traces ChatML** | [huggingface.co/datasets/sroecker/hermes-agent-traces-chatml](https://huggingface.co/datasets/sroecker/hermes-agent-traces-chatml) | Split `train` | Trajetórias agênticas do Nous Research Hermes com raciocínio prévio em `<think>` e chamadas de ferramentas de múltiplos turnos. |
| **PSStrikes Commands (PowerShell)** | [huggingface.co/datasets/dessertlab/PSStrikes-commands](https://huggingface.co/datasets/dessertlab/PSStrikes-commands) | `data/train-00000-of-00001-25809ad170aa95b5.parquet` | **25.809 pares** de linguagem natural e comandos PowerShell. *(Atenção: requer filtro estrito de segurança para eliminar comandos de exploit/malware)*. |
| **OpenClaw Coder Dataset** | [huggingface.co/datasets/bellfire/openclaw-coder-dataset](https://huggingface.co/datasets/bellfire/openclaw-coder-dataset) | Split `train` | Trajetórias do agente local OpenClaw para automação de workspace (`safe-exec`, `filesystem-context`, manipulação de arquivos). |
| **UltraData / ToolMind Curated** | Matriz local em `data/cleaned/` (baseada no UltraData Tool-Use) | `toolmind_graphsyn`<br>`synth_memory_20260608` | **16.261 amostras** focadas em passagem estrita de tipos em APIs técnicas, finanças, matemática e persistência de memória episódica. |

---

## 📂 2. Datasets e Artefatos Locais na Workspace

Caminhos locais dos dados limpos e bases de avaliação:

* **Matriz de Dados Curados (Fonte Primária):**
  * [`data/cleaned/ultradata_toolmind_curated_16k.jsonl`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/data/cleaned/ultradata_toolmind_curated_16k.jsonl) (46.95 MB, 16.261 amostras com 10.239 chamadas reais).
* **Dataset SFT Staging para Kaggle:**
  * [`Kaggle/kaggle_sft_data/curriculum_3000_tool_sft.jsonl`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/Kaggle/kaggle_sft_data/curriculum_3000_tool_sft.jsonl) (8.671.336 bytes, **1983 amostras V1** — o nome mantém `3000` por compatibilidade com o notebook/Kaggle; contagem real em `V1_MANIFEST.md`).
* **Bateria de Avaliação Holdout (90 Casos Cegos):**
  * [`data/eval_benchmark_battery_v2.jsonl`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/data/eval_benchmark_battery_v2.jsonl) (197 KB, 90 itens rigorosamente distribuídos em N1 Abstenção, N2 Discriminação e N3 Algorítmico).
* **Catálogo de Ferramentas:**
  * [`data/tool_catalog/`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/data/tool_catalog/) e espelho em [`Kaggle/kaggle_sft_data/tool_catalog/`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/Kaggle/kaggle_sft_data/tool_catalog/).

---

## ☁️ 3. Recursos e IDs na Nuvem Kaggle

Para deploy e treinamento nos nós Dual Tesla T4:

| Recurso | Slug / ID no Kaggle | Link no Kaggle | Função |
| :--- | :--- | :--- | :--- |
| **Dataset de Treino SFT** | `lordnyx0/qwen3loop-tool-sft-data` | [kaggle.com/datasets/lordnyx0/qwen3loop-tool-sft-data](https://www.kaggle.com/datasets/lordnyx0/qwen3loop-tool-sft-data) | Contém o arquivo `.jsonl` do dataset, a engine em `.tar` e o script de treino. |
| **Bundle do Modelo Base** | `lordnyx0/qwen3loop-tool-grpo-bundle` | [kaggle.com/datasets/lordnyx0/qwen3loop-tool-grpo-bundle](https://www.kaggle.com/datasets/lordnyx0/qwen3loop-tool-grpo-bundle) | Contém os pesos iniciais em Safetensors (`qwen3loop_sft_final`) e o tokenizer. |
| **Kernel Notebook de Treino** | `lordnyx0/qwen3loop-tool-sft` | [kaggle.com/code/lordnyx0/qwen3loop-tool-sft](https://www.kaggle.com/code/lordnyx0/qwen3loop-tool-sft) | Notebook de execução remota DDP Dual GPU Tesla T4. |

---

## 💻 4. Catálogo de Scripts Utilizados

A tabela abaixo mapeia cada script, sua localização e seu papel exato no ciclo de vida do projeto:

### A. Construção e Curadoria de Dados
* [`scripts/build_archive/build_curriculum_3000_sft.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scripts/build_archive/build_curriculum_3000_sft.py): Script mestre de extração anterior (serve de base para o novo construtor purificado).
* [`scratch/audits/audit_curriculum_3000.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scratch/audits/audit_curriculum_3000.py): Script de auditoria amostral profunda (inspeciona conformidade ChatML, `token_length`, `is_retrajectory` e exibe diálogos).
* [`scratch/audits/test_hermes_parse.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scratch/audits/test_hermes_parse.py): Validador de extração de chamadas `<tool_call>` do Hermes para `tool_calls` estruturados.
* [`scratch/audits/analyze_ultradata.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scratch/audits/analyze_ultradata.py): Analisador de fontes e distribuição de ferramentas do `ultradata_toolmind_curated_16k.jsonl`.

### B. Treinamento e Engine Recorrente
* [`Kaggle/kaggle_sft_data/train_sft_tool_qwen3loop.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/Kaggle/kaggle_sft_data/train_sft_tool_qwen3loop.py): Script de treinamento SFT Full-Parameter (PyTorch DDP, BitsAndBytes 8-bit AdamW, Multi-Exit Loss, Gradient Checkpointing e treino do Latent Halting Probe).
* [`engine/qwen3loop_python/modeling_qwen3loop.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/engine/qwen3loop_python/modeling_qwen3loop.py): Implementação da arquitetura Qwen3Loop (recorrência física de 28 camadas com virtualização de 56 camadas no KV Cache).
* [`engine/qwen3loop_python/configuration_qwen3loop.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/engine/qwen3loop_python/configuration_qwen3loop.py): Configuração e hiperparâmetros da arquitetura recorrente (`num_loops`, `loop_loss_weights`).

### C. Deploy e Monitoramento no Kaggle
* [`Kaggle/kaggle_kernel/qwen3loop_tool_sft.ipynb`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/Kaggle/kaggle_kernel/qwen3loop_tool_sft.ipynb): Notebook Jupyter executado nas Tesla T4 do Kaggle.
* [`Kaggle/kaggle_kernel/kernel-metadata.json`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/Kaggle/kaggle_kernel/kernel-metadata.json): Arquivo de configuração de fontes e acelerador do Kaggle.
* [`scratch/deploy/push_curriculum_3000.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scratch/deploy/push_curriculum_3000.py): Script de automação que obtém token OAuth e realiza push do dataset e do kernel.
* `scratch/monitoring/` + `kaggle kernels status lordnyx0/qwen3loop-tool-sft`: monitoramento dos logs remotos (nota: o `poll_logs` citado em versões antigas **não existe**).

### D. Unrolling, Quantização e Avaliação
* [`scripts/quantize/build_and_quantize_tool_sft_gguf.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scripts/quantize/build_and_quantize_tool_sft_gguf.py): Desdobra as 28 camadas físicas em 56 camadas lógicas e converte para GGUF quantizado em `Q8_0`.
* [`scripts/eval/benchmark_tool_calling_baseline.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scripts/eval/benchmark_tool_calling_baseline.py): Avaliação do checkpoint Safetensors contra os 90 casos cegos.
* [`scripts/eval/benchmark_tool_calling_q8.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scripts/eval/benchmark_tool_calling_q8.py): Avaliação do arquivo final GGUF Q8_0 no runtime de inferência.
