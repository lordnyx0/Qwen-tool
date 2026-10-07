# 05. Método de Treino SFT e Mecânica da Recorrência em Profundidade (Loop)

Este documento detalha o método formal de treinamento, a fundamentação matemática, o modelo base utilizado e a mecânica exata de como a **recorrência em profundidade (*layer looping*)** é tratada durante o Supervised Fine-Tuning (SFT) do **Qwen3Loop-0.6B**.

---

## 🧠 1. Modelo Base Utilizado

* **Identificador do Checkpoint Base:** [`checkpoints/qwen3loop_sft_final/`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/checkpoints/qwen3loop_sft_final)
* **Parâmetros Físicos:** 594.137.600 (~594M parâmetros reais).
* **Camadas Físicas no Grafo:** 28 camadas decodificadoras (`Qwen2DecoderLayer`).
* **Dimensão Oculta ($d_{\text{model}}$):** 1024.
* **Atenção:** 16 Query Heads, 8 Key/Value Heads (Grouped-Query Attention - GQA, proporção 2:1).
* **Dimensão Intermediária MLP:** 2816 (ativação SwiGLU).
* **Vocabulário:** 151.936 tokens (padrão Qwen2.5 ChatML).
* **Origem Genealógica:** Inicializado a partir dos pesos base de linguagem do Qwen2.5-0.5B e pré-alinhado na fase Base SFT para estabilização das representações recorrentes.

---

## 🔄 2. A Mecânica do Loop Recorrente no SFT

Diferente de transformers padrão (que processam a sequência através de uma pilha linear estática de camadas $1 \to 2 \to \dots \to N$), o **Qwen3Loop** reutiliza os pesos das camadas físicas ciclicamente dentro do mesmo passo autorregressivo.

### 2.1 Formulação Matemática da Recorrência

Seja $\mathbf{h}^{(0)}$ o embedding inicial da sequência após a camada de entrada. Para um número de passagens $K = \text{num\_loops} = 2$:

$$\mathbf{h}^{(k)} = \text{RMSNorm}\left( \mathcal{F}_{\Theta}\left( \mathbf{h}^{(k-1)} \right) \right), \quad k \in \{1, 2\}$$

Onde:
* $\mathcal{F}_{\Theta}$ representa o bloco unificado de 28 camadas físicas do transformador.
* $\Theta$ são os pesos compartilhados (*tied weights*): a camada física $i$ executa na passagem 1 e na passagem 2 sob o mesmo conjunto exato de tensores.
* $\mathbf{h}^{(1)}$ é o estado de saída intermediário ao fim do Loop 1.
* $\mathbf{h}^{(2)}$ é o estado de saída profundo ao fim do Loop 2.

```
                    [ Tokens de Entrada / Prompt + Turnos ]
                                     │
                                     ▼
                        Embeddings (Vocab 151.936)
                                     │
           ┌─────────────────────────┴─────────────────────────┐
           │                                                   │
           ▼ (Passagem 1: Loop 1)                              │
┌──────────────────────────────────────┐                       │
│  Camadas Físicas 0 a 27 (Pesos θ)    │                       │
└──────────────────────────────────────┘                       │
           │                                                   │
           ▼                                                   ▼
     RMSNorm Final                               Sonda Latente (Halting Probe)
           │                                     Prevê dificuldade do prompt
           ├─────────────────────────┐           [0: 1-loop | 1: 2-loops | 2: 3-loops]
           │                         ▼
           │             Loss Intermediária (w₁ = 0.4)
           ▼ (Passagem 2: Loop 2)
┌──────────────────────────────────────┐
│  Camadas Físicas 0 a 27 (Mesmos θ)   │  <── Pesos compartilhados (Tied Recurrence)
└──────────────────────────────────────┘
           │
           ▼
     RMSNorm Final
           │
           ▼
  Loss Final (w₂ = 1.0) ──> Projeção Linear lm_head ──> Logits / <tool_call>
```

---

## ⚙️ 3. Implementação Técnica da Engine (`modeling_qwen3loop.py`)

No treinamento supervisionado, a passagem de múltiplos loops impõe desafios técnicos severos que foram resolvidos na engine customizada:

### A. Indexação Virtual do KV Cache
Se a camada física 5 for executada no Loop 1 e depois no Loop 2, o KV Cache colidiria caso usasse o índice físico `layer_idx = 5`.
Para resolver isso, a engine calcula um **índice de camada virtual**:

$$\text{virtual\_layer\_idx} = \text{loop\_idx} \times \text{num\_hidden\_layers} + \text{orig\_layer\_idx}$$

* No Loop 1: Camadas 0 a 27 gravam nos slots de cache 0 a 27.
* No Loop 2: Camadas 0 a 27 gravam nos slots de cache 28 a 55.
O modelo aloca internamente um cache para **56 camadas virtuais**, garantindo separação absoluta dos tensores de atenção passados.

### B. Compatibilidade com Gradient Checkpointing (`use_reentrant=False`)
No PyTorch padrão, o checkpointing reentrante falha catastroficamente quando o mesmo módulo é chamado duas vezes no mesmo grafo de retropropagação. 
Por isso, a execução de cada camada foi encapsulada com:

```python
hidden_states = torch.utils.checkpoint.checkpoint(
    custom_forward,
    hidden_states,
    attn_mask,
    use_reentrant=False  # OBRIGATÓRIO para módulos recorrentes tied
)
```
O closure `custom_forward` preserva e restaura `virtual_layer_idx` e `orig_layer_idx` dinamicamente com bloco `try...finally`.

---

## 🎯 4. A Função de Perda: Multi-Exit Intermediate Loss

Para que o modelo não sofra colapso de representação intermediária e mantenha a capacidade de responder rapidamente em consultas simples, o treinamento supervisiona **todas as saídas de loop simultaneamente**.

### Formulação:
$$\mathcal{L}_{\text{total}} = \frac{w_1 \cdot \mathcal{L}_{\text{LM}}^{(1)} + w_2 \cdot \mathcal{L}_{\text{LM}}^{(2)}}{w_1 + w_2} + \lambda_{\text{probe}} \cdot \mathcal{L}_{\text{probe}}$$

### Hiperparâmetros Calibrados:
* $w_1 = 0.4$ (Peso da perda de linguagem no Loop 1).
* $w_2 = 1.0$ (Peso da perda de linguagem no Loop 2).
* $\lambda_{\text{probe}} = 0.15$ (Supervisão da cabeça classificadora de complexidade).

### 4.1 Máscara Estrita de Perda (`assistant_only_loss`)
No treinamento SFT de Tool Calling, o modelo **NÃO** deve calcular perda sobre:
* Tokens do System Prompt (`<|im_start|>system ... <|im_end|>`) $\to$ Rótulo `-100`.
* Tokens da Mensagem do Usuário (`<|im_start|>user ... <|im_end|>`) $\to$ Rótulo `-100`.
* Tokens das Observações de Ferramenta (`<|im_start|>tool ... <|im_end|>`) $\to$ Rótulo `-100`.

A perda de entropia cruzada incide **exclusivamente** sobre os tokens gerados pelo assistente:
1. Pensamento reflexivo: `<think> ... </think>`.
2. Emissão de chamada estruturada: `<tool_call> ... </tool_call>`.
3. Resposta e síntese final pós-execução da ferramenta.
4. Token delimitador de encerramento de turno: `<|im_end|>`.

### 4.2 Projeção Chunked de Logits (Cap de VRAM a 310 MB)
Projetar uma sequência de 5.000 tokens com vocabulário de 151.936 elementos consome $> 2.8\text{ GB}$ de VRAM em tensores de logits float32. 
A engine aplica o cálculo chunked em fatias de 1024 tokens ativos (`_compute_chunked_ce`), limitando o consumo de projeção a apenas **~310 MB** por GPU.

---

## 📡 5. A Sonda Latente: `LatentHaltingProbe`

A sonda é uma rede neural ultraleve acoplada em paralelo na saída da primeira iteração:

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

* **Entrada:** Vetor de ativação oculto do último token do prompt ao término do Loop 1 ($\mathbf{h}_{T_{\text{prompt}}}^{(1)}$).
* **Alvo de Classificação (`target_rec`):**
  * `Classe 0 (1 Loop / Early Exit)`: Consultas simples, abstenções diretas e chamadas de ferramenta de passo único.
  * `Classe 1 (2 Loops / Standard)`: Discriminação multi-ferramentas, parametrização estrita de APIs técnicas.
  * `Classe 2 (3 Loops / Deep Recurrence)`: Resolução de scripts complexos no terminal, retrajetórias corretivas após erros e raciocínio algorítmico.

---

## 📦 6. Do Treino Recorrente ao Unrolling em GGUF (56 Camadas)

Para inferência de produção em C++ (`llama.cpp`, Ollama, vLLM), grafos cíclicos dinâmicos não são suportados nativamente sem modificação profunda de binários.

O Qwen3Loop resolve isso através do **Unrolling Estático**:
1. O checkpoint treinado contém 28 camadas físicas salvas em `model.safetensors`.
2. O script [`scripts/quantize/build_and_quantize_tool_sft_gguf.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scripts/quantize/build_and_quantize_tool_sft_gguf.py) desdobra as 28 camadas físicas em **56 camadas lógicas**:
   * Camada $0 \dots 27$ no GGUF $\leftarrow$ Pesos físicos do Loop 1.
   * Camada $28 \dots 55$ no GGUF $\leftarrow$ Mesmos pesos físicos mapeados para o Loop 2.
3. O modelo resultante é quantizado em **GGUF Q8_0 (1.11 GB)**, comportando-se externamente como um modelo padrão Qwen de 56 camadas lógicas, executável com aceleração máxima de hardware sem necessitar de runtimes customizados.
