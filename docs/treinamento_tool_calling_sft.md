# Treinamento Supervisionado (SFT): Tool-Calling no Qwen3Loop

Este documento descreve a metodologia e execução do treinamento supervisionado **Full-Parameter SFT** do **Qwen3Loop-0.6B** especializado em **Tool Calling**.

---

## 1. Justificativa Técnica: Por que SFT e não LoRA/RL?

Durante os experimentos com Reinforcement Learning (**RIC-GRPO**) e LoRA ($r=16$), identificamos dois obstáculos fundamentais:
1. **Aniquilação por Quantização:** Os deltas aprendidos via LoRA com $\text{LR}=2 \times 10^{-5}$ produziam perturbações de peso $\Delta W \approx 10^{-4}$. Ao quantizar o modelo para GGUF Q8_0 ($\Delta_{Q8} \approx 2.7 \times 10^{-3}$), o arredondamento numérico apagava mais de 98% dos gradientes de LoRA fundidos.
2. **Problema do Início a Frio (Zero-Gradient Exploration):** O algoritmo de RL precisava gerar código Python sintaticamente correto a partir de amostragem estocástica pura. Sem demonstrações prévias, o modelo nunca atingia o resultado de execução correto em tarefas de Nível 3, gerando recompensas zero e gradientes nulos em mais de 60% dos passos.

O **Full-Parameter SFT** resolve ambos os problemas:
- Aplica gradientes densos de Cross-Entropy diretamente sobre os 594M parâmetros físicos.
- Modifica as matrizes de peso em magnitudes macroscopicamente superiores ao degrau de quantização de 8 bits, tornando o comportamento **imune à degradação na conversão GGUF Q8_0 e Q4_K_M**.

---

## 2. Configuração de Hiperparâmetros

| Hiperparâmetro | Valor | Descrição |
| :--- | :--- | :--- |
| **Taxa de Aprendizado ($\eta$)** | `2e-5` | Otimização suave para evitar esquecimento catastrófico |
| **Otimizador** | `bitsandbytes.optim.AdamW8bit` | Estado do otimizador reduzido para 1.19 GB |
| **Batch Size Físico** | `2` | Batch de sequências por GPU |
| **Gradient Accumulation** | `8` | Passos de acumulação |
| **Batch Efetivo** | **16** | $2 \times 8 = 16$ amostras por atualização de pesos |
| **Épocas** | `2` | ~384 passos de otimizador para 3.071 amostras |
| **Comprimento Máximo ($L_{\max}$)** | `1024` tokens | Cobre prompts de schemas complexos do Hermes |
| **Multi-Exit Weights** | `[0.4, 1.0]` | Perda ponderada na passagem intermediária e final |
| **Peso da Sonda Latente** | `0.15` | Cross-Entropy supervisionada da sonda de halting |
| **Precisão** | PyTorch Nativo `bfloat16` / `float16` | Aceleração por Tensor Cores |
| **Gradient Checkpointing** | Ativado | Recálculo de ativações para economizar VRAM |

---

## 3. Pipeline de Execução no Kaggle

O treinamento foi configurado para execução distribuída ou acelerada na nuvem Kaggle via Kaggle CLI:

### 3.1 Datasets Vinculados:
1. `lordnyx0/qwen3loop-tool-grpo-bundle`: Pesos base (`model.safetensors`, 1.19 GB) + engine recorrente.
2. `lordnyx0/qwen3loop-tool-sft-data`: Dataset limpo de 3.071 amostras (`sft_tool_calling_curated.jsonl`, 5.04 MB) + script de treino.

### 3.2 Comando de Execução:
```bash
python train_sft_tool_qwen3loop.py \
    --model_path /kaggle/input/qwen3loop-tool-grpo-bundle/model \
    --data_path /kaggle/input/qwen3loop-tool-sft-data/sft_tool_calling_curated.jsonl \
    --output_dir /kaggle/working/qwen3loop_tool_sft_final \
    --batch_size 2 \
    --grad_accum 8 \
    --epochs 2 \
    --lr 2e-5 \
    --max_len 1024 \
    --eval_steps 50 \
    --save_steps 150
```

---

## 4. Estrutura do Checkpoint de Saída

Ao término do treinamento, o diretório salvo contém:
- `model.safetensors`: Matrizes de peso de todos os 594M parâmetros físicos atualizados.
- `latent_halting_probe.pt`: Pesos ajustados da cabeça MLP de previsão de complexidade.
- `config.json`, `tokenizer.json`, `tokenizer_config.json`: Metadados e vocabulário.
- `chat_template.jinja`: Template de formatação Qwen ChatML com injeção automática de `<think>` e `<tool_call>`.
