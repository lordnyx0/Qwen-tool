# 03. Pipeline de Execução e Infraestrutura Kaggle (Dual Tesla T4)

Este documento descreve detalhadamente a infraestrutura de computação em nuvem, as otimizações de memória VRAM e o fluxo de deploy contínuo para treinar o **Qwen3Loop-0.6B** no Kaggle sem risco de OOM ou incompatibilidade de hardware.

---

## 🖥️ 1. Ambiente Computacional

* **Aceleradores:** 2x NVIDIA Tesla T4 (Arquitetura Turing, 16 GB VRAM por placa, Compute Capability 7.5).
* **Topologia de Treino:** PyTorch Distributed Data Parallel (DDP) orquestrado pelo Hugging Face `accelerate`.
* **Processos:** 2 workers (`--num_processes 2`), um por GPU.

---

## ⚡ 2. Otimizações Críticas de Memória e VRAM

O modelo possui 594M parâmetros físicos. Sem otimização, o otimizador AdamW clássico em FP32 consumiria 4.75 GB apenas em estados de momento, correndo alto risco de OOM durante o cálculo de gradientes de sequências longas (até 5.000 tokens). 

As seguintes otimizações estão ativas no script [`Kaggle/kaggle_sft_data/train_sft_tool_qwen3loop.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/Kaggle/kaggle_sft_data/train_sft_tool_qwen3loop.py):

1. **Otimizador BitsAndBytes AdamW 8-bit (`bnb.optim.AdamW8bit`):**
   * Reduz o footprint de memória dos momentos do otimizador de 4.75 GB para **apenas 1.19 GB** por GPU.
2. **Gradient Checkpointing Não-Reentrante:**
   * Implementado em [`engine/qwen3loop_python/modeling_qwen3loop.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/engine/qwen3loop_python/modeling_qwen3loop.py) via `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
   * Preserva intacto o estado latente intermediário `h_1` para alimentar a cabeça supervisora `LatentHaltingProbe`.
3. **Scaled Dot-Product Attention (SDP) para Turing:**
   * A arquitetura Turing (T4) não possui suporte nativo a FlashAttention-2.
   * O script força o kernel de memória eficiente Cutlass:
     ```python
     torch.backends.cuda.enable_mem_efficient_sdp(True)
     torch.backends.cuda.enable_flash_sdp(False)
     torch.backends.cuda.enable_math_sdp(True)
     ```
4. **Precisão FP16 Estrita:**
   * Os Tensor Cores da Tesla T4 aceleram nativamente FP16 (não BF16). O pipeline opera sob `torch.float16` nativo em autocast.

---

## 📦 3. Empacotamento e Publicação do Dataset

O Kaggle possui uma limitação de upload de múltiplos arquivos pequenos (gerando erros ou lentidão). Para contornar isso, o repositório utiliza empacotamento em `.tar`:

```bash
# Estrutura do diretório Kaggle/kaggle_sft_data/
Kaggle/kaggle_sft_data/
├── curriculum_3000_tool_sft.jsonl     # Dataset SFT principal
├── dataset-metadata.json              # Metadados do dataset no Kaggle
├── qwen3loop_python/                  # Engine arquitetural (empacotada como .tar)
├── tool_catalog/                      # Esquemas das ferramentas
└── train_sft_tool_qwen3loop.py        # Script de treinamento SFT
```

### Comando de Publicação/Atualização do Dataset:
O upload deve sempre utilizar a flag `-r tar` para compactar automaticamente subdiretórios:
```bash
kaggle datasets version -p Kaggle/kaggle_sft_data -m "Versão Limpa SFT" -r tar
```

*(Caso seja a primeira criação do dataset, utilize `kaggle datasets create -p Kaggle/kaggle_sft_data -r tar`)*.

---

## 🚀 4. Configuração do Notebook Kernel e Deploy

O notebook executado no Kaggle reside em [`Kaggle/kaggle_kernel/qwen3loop_tool_sft.ipynb`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/Kaggle/kaggle_kernel/qwen3loop_tool_sft.ipynb) e seus metadados em [`Kaggle/kaggle_kernel/kernel-metadata.json`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/Kaggle/kaggle_kernel/kernel-metadata.json).

### Metadados (`kernel-metadata.json`):
```json
{
  "id": "lordnyx0/qwen3loop-tool-sft",
  "title": "qwen3loop-tool-sft",
  "code_file": "qwen3loop_tool_sft.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "enable_internet": true,
  "dataset_sources": [
    "lordnyx0/qwen3loop-tool-grpo-bundle",
    "lordnyx0/qwen3loop-tool-sft-data"
  ]
}
```

### Autenticação e Disparo do Kernel via CLI:
Para autenticar na CLI do Kaggle utilizando o token OAuth localmente instalado, use o padrão do script de automação:
```python
import subprocess, os

# Obter token de acesso ativo
token = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = token
env['PYTHONIOENCODING'] = 'utf-8'

# Enviar kernel atualizado para execução imediata
subprocess.run(['kaggle', 'kernels', 'push', '-p', 'kaggle_kernel'], env=env)
```

---

## 📊 5. Monitoramento Remoto de Logs

Para acompanhar o progresso do treinamento no Kaggle sem abrir o navegador:
```bash
kaggle kernels status lordnyx0/qwen3loop-tool-sft
```
(Nota: o `poll_logs` citado em versões antigas **não existe**; monitores legados vivem em `scratch/monitoring/`.)
Esse utilitário lê o stream de execução da API do Kaggle e filtra linhas de progresso do PyTorch DDP em tempo real.
