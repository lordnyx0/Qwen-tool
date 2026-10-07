# Arquitetura Recorrente: Qwen3Loop-0.6B

Este documento detalha o design de engenharia, a mecânica de inferência e a estrutura de tensores da arquitetura **Qwen3Loop-0.6B**.

---

## 1. Visão Geral do Design

O **Qwen3Loop** é uma implementação especializada de um **Transformer Recorrente em Profundidade (Depth-Recurrent Transformer)** baseado na família Qwen2.5 / Qwen3. Em vez de adicionar camadas físicas adicionais (o que aumentaria a contagem de parâmetros e a pegada de VRAM), o modelo mantém uma pilha física compacta de **28 camadas** e itera ciclicamente através delas.

```
                           [Token Embedding]
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │  Camadas Físicas 0 a 6       │ (Prefixo Estático)
                   └──────────────┬───────────────┘
                                  │
                ┌─────────────────┴─────────────────┐
                │                                   ▼
                │                    ┌──────────────────────────────┐
                │                    │  Camadas Físicas 7 a 20      │ <──┐
                │                    │  (Miolo de Raciocínio)       │    │ Loop 2
                │                    └──────────────┬───────────────┘    │
                │                                   ▼                    │
                │                           Saída do Loop 1 ─────────────┘
                │                                   │
                ▼                                   ▼
    ┌───────────────────────┐        ┌──────────────────────────────┐
    │ Latent Halting Probe  │        │  Camadas Físicas 21 a 27     │ (Sufixo Estático)
    │ [Decisão de Saída]    │        └──────────────┬───────────────┘
    └───────────────────────┘                       │
                                                    ▼
                                            [RMSNorm & LM Head]
```

---

## 2. Especificações do Modelo Físico

| Parâmetro | Valor | Detalhes |
| :--- | :--- | :--- |
| **Parâmetros Físicos** | **594.041.856** (~594M) | 100% preservados na memória |
| **Camadas Físicas** | **28** (`Qwen2DecoderLayer`) | Camadas reais instanciadas |
| **Dimensão Oculta ($d_{model}$)** | **1024** | Dimensão dos estados latentes |
| **Dimensão Intermediária (MLP)** | **2816** | Projeções `gate_proj`, `up_proj`, `down_proj` |
| **Attention Heads (Q)** | **16** | Dimensão da cabeça $d_k = 64$ |
| **KV Heads (K, V)** | **8** | Grouped-Query Attention (GQA 2:1) |
| **Tamanho do Vocabulário** | **151.936** | Tokenizer Qwen2.5 ChatML com tags especiais |
| **Precisão Nativa** | `bfloat16` / `float16` | 1.19 GB em pesos FP16 |

---

## 3. Indexação Virtual do KV Cache

Um dos maiores desafios técnicos de transformadores com loop é a alocação de chaves e valores (**KV Cache**). Se o mesmo token visitar a camada física $i$ na iteração 1 e na iteração 2, os tensores K e V não podem sobrescrever a mesma posição de memória sem corromper a atenção causal.

No `Qwen3Loop`, implementou-se o sistema de **KV Cache com Indexação Virtual**:

$$\text{virtual\_layer\_idx} = i + (\text{loop\_idx} \times N_{\text{physical\_layers}})$$

- Na primeira iteração ($\text{loop\_idx} = 0$), a camada física $i \in [0, 27]$ escreve nos slots virtuais $[0, 27]$.
- Na segunda iteração ($\text{loop\_idx} = 1$), a camada física $i \in [0, 27]$ escreve nos slots virtuais $[28, 55]$.
- O KV Cache aloca dinamicamente até **56 camadas virtuais**, permitindo que a atenção na segunda passagem atenda tanto aos tokens originais quanto aos estados transformados da primeira passagem.

---

## 4. Sonda Latente de Halting (Latent Halting Probe)

Para conferir adaptabilidade computacional (*Adaptive Computation Time - ACT*), o modelo incorpora uma cabeça de sonda ultraleve conectada à saída da primeira iteração:

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

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_state)
```

- **Número de Parâmetros:** ~65.600 (~0.01 ms de inferência em GPU).
- **Entrada:** Vetor latente do último token de prompt após a primeira iteração ($\mathbf{h}^{(1)}_{T}$).
- **Classes de Saída:**
  - `0 (Raso)`: Consultas factuais simples ou abstenções conversacionais (1 loop suficiente).
  - `1 (Médio)`: Chamada de função padrão com seleção de schema (2 loops recomendados).
  - `2 (Profundo)`: Código Python algorítmico, lógica matemática ou schemas aninhados complexos (2+ loops obrigatórios).

---

## 5. Arquitetura de Arquivos da Engine

A implementação da engine está isolada no diretório [`engine/qwen3loop_python`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/engine/qwen3loop_python):

- **`configuration_qwen3loop.py`**: Declara a classe `Qwen3LoopConfig`, registrando parâmetros customizados como `num_loops`, `loop_loss_weights`, `enable_double_loop_split` e `skip_loop_final_norm`.
- **`modeling_qwen3loop.py`**: Subclasse nativa de `Qwen3Model` e `Qwen3ForCausalLM`. Contém o loop de execução explícito, manipulação de causal mask e cálculo da perda multi-exit ponderada.
- **`__init__.py`**: Registra as classes nos mappings `AutoConfig` e `AutoModelForCausalLM` do Hugging Face.
