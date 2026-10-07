# Metodologia de Engenharia e Treinamento: Qwen3Loop-0.6B (Base SFT)

Este documento descreve detalhadamente o método, formulação matemática, arquitetura recorrente, engenharia de memória e pipeline de dados utilizados no primeiro treinamento supervisionado (**Base SFT**) do **Qwen3Loop-0.6B**.

---

## 1. Visão Geral e Filosofia do Modelo

O **Qwen3Loop-0.6B** é uma arquitetura de modelo de linguagem de raciocínio compacto (594M parâmetros físicos) projetada sob o paradigma de **Recorrência em Profundidade (Depth Recurrence / Layer Looping)**.

O objetivo do treino base foi instilar capacidade analítica densa (*Chain-of-Thought*) em um modelo ultraleve, fazendo com que blocos intermediários de transformadores fossem reutilizados ciclicamente em um mesmo passo autorregressivo:

$$\mathbf{h}^{(k)} = \mathcal{F}_{\theta}(\mathbf{h}^{(k-1)}), \quad k \in \{1, 2\}$$

Onde $\mathcal{F}_{\theta}$ representa as camadas físicas do transformador com pesos amarrados (*tied weights*), e $k$ é o índice de iteração (loop).

```
                      [Entrada Tokenizada]
                               │
                               ▼
                    Camadas Iniciais 0 a 6
                               │
                ┌──────────────┴──────────────┐
                │                             ▼
                │                  Camadas Centrais 7 a 20  <──┐
                │                             │                │ (Loop 2)
                │                             ▼                │
                │                  Saída do Miolo Recorrente ──┘
                │                             │
                ▼                             ▼
       Sonda Latente (Halting)      Camadas Finais 21 a 27
       [Loop 1 Check: Early Exit?]            │
                                              ▼
                                       Logits / Saída
```

---

## 2. Topologia da Arquitetura Recorrente

### 2.1 Estrutura de Camadas Físicas vs. Lógicas
- **Camadas Físicas:** 28 camadas (`Qwen2DecoderLayer`).
- **Camadas Lógicas no Treino:** 56 camadas virtuais geradas por 2 iterações consecutivas (`num_loops = 2`).
- **Dimensão Oculta ($d_{model}$):** 1024
- **Attention Heads:** 16 Q-heads, 8 KV-heads (Grouped-Query Attention - GQA).
- **Vocabulário:** 151.936 tokens (Qwen2.5 ChatML).

### 2.2 Unrolling para Inferência Padrão (llama.cpp / GGUF)
Em inferência otimizada local, engines como o `llama.cpp` esperam grafos acíclicos estáticos. Para viabilizar a execução de alta velocidade sem reescrever o kernel C++ do runtime, as 28 camadas físicas são **desenroladas (*unrolled*)** mapeando cada camada física $l$ na camada lógica $l + 28$:
- Camada Física $i$ $\to$ Camadas Lógicas $i$ e $i + 28$.
- Os tensores são duplicados nos nós do arquivo GGUF apontando para a mesma estrutura lógica, transformando o modelo em um **Qwen padrão de 56 camadas lógicas**.

---

## 3. Função de Perda: Multi-Exit Intermediate Loss

Um dos desafios fundamentais do treinamento recorrente é o colapso de gradientes e a divergência entre passagens intermediárias: se o modelo só for otimizado na saída da última iteração, as ativações intermediárias degradam-se ou geram representações caóticas.

Para solucionar isso, implementou-se a **Multi-Exit Intermediate Loss**:

$$\mathcal{L}_{\text{total}} = w_1 \cdot \mathcal{L}_{\text{loop1}} + w_2 \cdot \mathcal{L}_{\text{loop2}} + \lambda_{\text{probe}} \cdot \mathcal{L}_{\text{probe}}$$

### Parâmetros Definidos no Treino:
- $w_1 = 0.4$ (Penalidade da passagem intermediária na iteração 1)
- $w_2 = 1.0$ (Penalidade da passagem final na iteração 2)
- $\lambda_{\text{probe}} = 0.15$ (Supervisão da cabeça de sonda latente)

### Formulação da Perda de Linguagem (LM Loss):
Para cada saída de loop $k \in \{1, 2\}$, a perda de entropia cruzada autorregressiva é calculada exclusivamente sobre os tokens de resposta do assistente:

$$\mathcal{L}_{\text{loop}_k} = -\frac{1}{|\mathcal{T}_{\text{resp}}|} \sum_{t \in \mathcal{T}_{\text{resp}}} \log P(x_t \mid x_{<t}, \text{loop}=k)$$

Isso força as camadas físicas a aprenderem transformações que já sejam semanticamente estruturadas no Loop 1 ($w_1=0.4$), enquanto refinam o raciocínio profundo no Loop 2 ($w_2=1.0$).

---

## 4. Supervisão Latente: Latent Halting Probe

Para permitir que o modelo decida quando parar o processamento (Early-Exit Dinâmico), foi acoplada uma sonda latente ultraleve:

### Arquitetura da Sonda:
```python
class LatentHaltingProbe(nn.Module):
    def __init__(self, hidden_size: int = 1024, num_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 64, bias=True),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes, bias=False)
        )
```

- **Tamanho:** ~65.000 parâmetros (~0.01 ms de latência).
- **Entrada:** Vetor de ativação do último token de prompt na saída do Loop 1 ($\mathbf{h}_{T_{\text{prompt}}}^{(1)}$).
- **Classes de Saída:**
  - `0`: 1 iteração necessária (raciocínio raso / consulta factual).
  - `1`: 2 iterações necessárias (raciocínio padrão / decomposição).
  - `2`: 3+ iterações (raciocínio analítico complexo / algoritmos).
- **Perda da Sonda:** Cross-Entropy supervisionada pelos metadados cognitivos gerados pelo Laya (`target_rec`).

---

## 5. Curadoria e Meta-Supervisão de Dados (Laya Engine)

A qualidade do dataset foi o divisor de águas para que o modelo de 0.6B aprendesse a raciocinar sem alucinar:

### 5.1 Pipeline de Purificação
O dataset bruto foi filtrado a partir de fontes de alta pureza (`PRIME-RL`, `Eurus-2`, `Open-R1 / Codeforces`):
1. **Verificação Estrutural:** Validação estrita de fechamento de blocos de código Markdown (```).
2. **Janela de Tamanho:** 300 a 5.500 caracteres (descarte de respostas triviais ou prolixas).
3. **Filtro de Entropia:** Entropia de 4-gramas $> 0.68$ (eliminação de loops degenerativos de repetição de texto).

### 5.2 Anotação Cognitiva com `convaiinnovations/laya`
O modelo Laya analisou 6.324 amostras, classificando a assinatura cognitiva de cada uma:
- **Estratégias Identificadas:**
  - 56.4% Analíticas (dedução passo a passo)
  - 28.3% Agênticas (estruturas com tags explícitas `[ASSESS]`, `[VERIFY]`, `[EXECUTE]`)
  - 15.3% Algorítmicas (código com análise de complexidade $O(N)$)
- **Recorrência:** Amostras com raciocínio profundo receberam rótulo de supervisão para o Halting Probe (`recurrence_value = 2` ou `3`).

---

## 6. Engenharia de Memória e Otimização

Para permitir **Full Fine-Tuning (todos os 594M parâmetros livres)** sem estourar os 12 GB da GPU NVIDIA RTX 3060 local:

| Componente | Configuração Adotada | Impacto no Treino |
| :--- | :--- | :--- |
| **Otimizador** | `bitsandbytes.optim.AdamW8bit` | Redução do estado do otimizador de **4.8 GB para 1.19 GB** |
| **Precisão** | PyTorch nativo `bfloat16` (`torch.cuda.is_bf16_supported()`) | Prevenção de underflow/overflow numérico sem perda de gradiente |
| **Gradient Checkpointing** | `model.gradient_checkpointing_enable()` | Recálculo de ativações nas camadas intermediárias |
| **Batching Efetivo** | Batch físico 1 + Gradient Accumulation 16 | Batch efetivo de 16 sequências, mantendo estabilidade estocástica |
| **VRAM Total em Pico** | **5.58 GB** | Mais de 6 GB de margem livre na VRAM |

---

## 7. Estratégia de Máscara de Perda (ChatML Masking)

Para evitar que o modelo desperdiçasse capacidade tentando memorizar o texto do usuário ou prompts do sistema, aplicou-se uma máscara rigorosa sobre os IDs dos tokens:

```python
prompt_txt = f"<|im_start|>system\n{system_txt}<|im_end|>\n<|im_start|>user\n{user_txt}<|im_end|>\n<|im_start|>assistant\n"
full_txt = prompt_txt + asst_txt + "<|im_end|>"

labels = list(full_ids)
# Mascarar todos os tokens de sistema e usuário com -100 (ignorado pelo CrossEntropyLoss)
for i in range(len(prompt_ids)):
    labels[i] = -100
```

Desta forma, os gradientes retropropagados decorrem **100% da cadeia de raciocínio e da resposta do assistente**.

---

## 8. Resultados Obtidos na Época 1

O treinamento executou 375 passos de otimizador (6.000+ amostras processadas) com convergência homogênea:

- **LM Loss Inicial:** `0.7075` $\to$ **LM Loss Final:** `0.4009`
- **Validation Loss:** `0.6705` $\to$ `0.4012`
- **Perplexidade:** `1.96` $\to$ **`1.49`**
- **Acurácia da Sonda de Halting:** Atingiu **88.4%** na predição da necessidade de iteração.
- **Sobrevivência à Quantização:** O modelo unrolled gerou um arquivo GGUF Q8_0 de **1.11 GB** que manteve a integridade estrutural e respostas completas no runtime `llama.cpp`.

---

## 9. Por que Esse Método Funciona Para o Tool-Calling SFT

1. **Alteração Macroscópica de Pesos:** Diferente do LoRA ($\Delta W \approx 10^{-4}$), o Full-FT atualiza os pesos principais de forma densa. Ao quantizar em Q8_0 ($\Delta \approx 10^{-3}$), os novos comportamentos (ex: emitir tags `<tool_call>` com argumentos JSON precisos) não são eliminados pelo arredondamento dos inteiros de 8 bits.
2. **Coerência Multi-Exit:** Ao manter a perda intermediária com peso $0.4$, o modelo mantém as representações internas alinhadas entre o Loop 1 e o Loop 2, permitindo que o unrolling para 56 camadas funcione perfeitamente sem degradar a geração de ferramentas.
3. **Capacidade Mista:** O mesmo pipeline permite mesclar tarefas de chamada de ferramentas com exemplos de raciocínio analítico puro, gerando um modelo que sabe pensar dentro da tag `<think>` antes de emitir a ferramenta `<tool_call>`.
