# Análise Post-Mortem: RIC-GRPO, LoRA e o Efeito da Quantização

Este documento detalha o estudo pós-treinamento da tentativa de Reinforcement Learning (**RIC-GRPO**) no Kaggle, explicando analiticamente por que o LoRA é aniquilado pela quantização de 8 bits e por que a transição para Full Fine-Tuning supervisionado (SFT) é necessária.

---

## 1. O Treinamento RIC-GRPO no Kaggle

Concluímos com sucesso a execução de **100 passos densos de RL distribuído** em duas GPUs Tesla T4 (32 GB VRAM combinada) usando o algoritmo GRPO (Group Relative Policy Optimization):
- **Passos Concluídos:** 100/100
- **Total de Rollouts:** 600 gerações estocásticas completas ($G = 6$ candidatos por prompt).
- **Tempo Total:** ~100 minutos.
- **Checkpoints Salvos:** Checkpoints a cada 25 passos e adaptadores finais consolidados (diretório da época, removido em limpeza posterior; adapters v4/v5 preservados em `Kaggle/kaggle_kernel_output/`).

---

## 2. Diagnóstico da Telemetria: O Fenômeno de "Passos com Gradiente Zero"

A análise computacional dos logs detalhados do treino revelou a seguinte distribuição de recompensas:

- **Passos com Recompensa Zero ($R = 0.0$):** **62 de 100 passos (62.0%)**.
- **Passos com Recompensa Positiva ($R > 0.0$):** **38 de 100 passos (38.0%)**.

### Análise Matemática:
No algoritmo GRPO, as vantagens normalizadas são calculadas como:

$$A_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}$$

Quando todos os $G=6$ candidatos tiram nota zero ($R_i = 0, \forall i \in \{1, \dots, 6\}$):
1. A média é $\bar{R} = 0.0$ e a variância é $\sigma_R = 0.0$.
2. As vantagens normalizadas são identicamente nulas ($A_i = 0.0, \forall i$).
3. A perda da política é:
   $$\mathcal{L}_{\text{policy}} = \sum_{i=1}^G A_i \cdot \log \pi_{\theta}(x_i) = 0.0$$
4. **O gradiente da política é o vetor nulo:** $\nabla_{\theta} \mathcal{L} = \vec{0}$.
5. Nenhum aprendizado ocorre nesses passos!

### Causa Raiz:
O modelo não possuía exemplos supervisionados de `execute_python`. Diante de problemas matemáticos de Nível 3, o modelo tentava inventar saídas textuais ou código truncado, nunca acertando a execução exata na amostragem aleatória.

---

## 3. A Matemática da Aniquilação do LoRA na Quantização Q8_0

O modelo treinado em GRPO foi fundido (*merged*) e quantizado para **GGUF Q8_0**. No entanto, a acurácia no teste foi exatamente igual à do modelo base original (52.2%).

Para investigar o motivo, medimos a magnitude numérica da matriz de perturbação de pesos:

$$\Delta W = \frac{\alpha}{r} (B \cdot A)$$

Com os parâmetros utilizados no GRPO ($r=16, \alpha=32, \text{LR}=2 \times 10^{-5}$):
- **Norma Média dos Pesos Base:** $\|W_{\text{base}}\| \approx 0.035$
- **Magnitude Máxima do Delta LoRA:** $|\Delta W| \approx 1.37 \times 10^{-4}$
- **Degrau de Escala da Quantização Q8_0:**
  $$\Delta_{Q8} = \frac{\max(|W|) - \min(|W|)}{255} \approx \frac{0.70}{255} \approx 2.75 \times 10^{-3}$$

### Comparação de Escalas:
$$\frac{\Delta_{Q8}}{|\Delta W|} \approx \frac{2.75 \times 10^{-3}}{1.37 \times 10^{-4}} \approx 20.07$$

O degrau de quantização de 8 bits é **20 vezes maior** que a atualização aprendida pelo LoRA.

Na operação de quantização de ponto flutuante para inteiro de 8 bits:

$$q = \text{round}\left(\frac{W + \Delta W}{\Delta_{Q8}}\right)$$

Como $|\Delta W| \ll \Delta_{Q8}$, para mais de **98% dos elementos da matriz**, tem-se:

$$\text{round}\left(\frac{W + \Delta W}{\Delta_{Q8}}\right) \equiv \text{round}\left(\frac{W}{\Delta_{Q8}}\right)$$

O arredondamento inteiro **eliminou a atualização do LoRA**, restaurando essencialmente os pesos do modelo pré-treinado original!

---

## 4. Conclusão e Resolução Estratégica

> *"Um modelo que não pode ser quantizado não é útil na prática."*

Para que as novas habilidades de chamada de ferramentas e execução de código sobrevivam à quantização **GGUF Q8_0** e **Q4_K_M**, as alterações nos pesos precisam ser macroscópicas e profundas.

**A solução definitiva é o Full Fine-Tuning (SFT de Parâmetros Livres):**
- Todos os 594M parâmetros físicos são otimizados diretamente.
- O gradiente de Cross-Entropy com máscara ChatML desloca os pesos em magnitude suficiente para alterar os índices inteiros quantizados permanentemente.
- Utilizando `bnb.optim.AdamW8bit`, o Full-FT consome apenas **~5.5 GB VRAM**, executando perfeitamente no Kaggle ou na RTX 3060 local.
