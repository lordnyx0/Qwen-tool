# Unrolling e Quantização GGUF: Qwen3Loop

Este documento descreve o processo de conversão, duplicação lógica (*unrolling*) e quantização em formato GGUF para inferência em alta velocidade no runtime `llama.cpp` e LM Studio / Ollama.

---

## 1. Por que Unrolling?

O modelo físico do **Qwen3Loop-0.6B** possui **28 camadas físicas**, mas é treinado para realizar duas passagens recorrentes (`num_loops = 2`).

Runtimes de inferência convencionais de alto desempenho (como `llama.cpp`) executam grafos de computação estáticos e lineares. Eles não possuem primitivas nativas para laços de repetição sobre a mesma camada com alocação virtual de KV Cache.

Para viabilizar a execução nativa sem necessidade de modificação no código-fonte em C++ do `llama.cpp`:
- As 28 camadas físicas são **desenroladas (*unrolled*)** para **56 camadas lógicas**.
- O modelo resultante passa a ser um modelo padrão compatível com a arquitetura `qwen2` / `qwen3` de 56 camadas.

```
Camadas Físicas (Treino PyTorch):
[0, 1, 2, ..., 27] ──(Iteração 1)──> [0, 1, 2, ..., 27] ──(Iteração 2)

Camadas Lógicas (GGUF Unrolled):
[0, 1, ..., 27]                      [28, 29, ..., 55]
      │                                     │
      └──────────── Mesmos Pesos ───────────┘
```

---

## 2. Pipeline de Unrolling Passo a Passo

O script de unrolling ([`scripts/quantize/build_and_quantize_grpo_gguf.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scripts/quantize/build_and_quantize_grpo_gguf.py); nota: `unroll_gguf.py` na raiz citado em versões antigas **não existe**) realiza as seguintes operações:

### Passo 1: Leitura dos Metadados Originais
- Carrega o arquivo GGUF original de 28 camadas (exportado via script oficial `convert_hf_to_gguf.py`).
- Lê a contagem de camadas no header GGUF:
  ```python
  qwen2.block_count = 28
  ```

### Passo 2: Duplicação Lógica dos Tensores
- Para cada tensor de peso de camada (`blk.X.attn_q.weight`, `blk.X.mlp_gate.weight`, etc.), o script cria uma cópia lógica mapeada para o índice da segunda iteração:
  ```python
  new_name = tensor_name.replace(f"blk.{i}.", f"blk.{i + 28}.")
  ```
- Atualiza o campo de configuração no cabeçalho do arquivo GGUF:
  ```python
  qwen2.block_count = 56
  ```

### Passo 3: Geração do Arquivo GGUF F16 Unrolled
- O resultado é salvo como `unrolled_modelo_qwen3loop_sft_f16.gguf` (~2.08 GB).
- Ele é 100% compatível com qualquer versão do `llama-cli`, `llama-server` ou engines derivadas.

---

## 3. Quantização para Q8_0 e Q4_K_M

A partir do arquivo F16 unrolled, utilizamos o utilitário nativo `llama-quantize`:

```bash
# Quantização para Q8_0 (Precisão de 8 bits balanceada)
llama-quantize.exe unrolled_modelo_qwen3loop_sft_f16.gguf \
                   models/gguf/unrolled_modelo_qwen3loop_sft_q8_0.gguf Q8_0

# Quantização para Q4_K_M (Otimização máxima de velocidade / VRAM)
llama-quantize.exe unrolled_modelo_qwen3loop_sft_f16.gguf \
                   models/gguf/unrolled_modelo_qwen3loop_sft_q4_k_m.gguf Q4_K_M
```

### Tamanho e Desempenho Comparativo:

| Formato | Tamanho em Disco | Consumo de RAM/VRAM | Velocidade de Inferência | Perda de Qualidade |
| :--- | :--- | :--- | :--- | :--- |
| **F16 Unrolled** | `2.08 GB` | ~2.5 GB | ~80 tokens/s | Zero (Referência) |
| **Q8_0 Unrolled** | **`1.11 GB`** | **~1.4 GB** | **~125 tokens/s** | **Praticamente imperceptível** |
| **Q4_K_M Unrolled** | **`0.68 GB`** | **~0.9 GB** | **~170 tokens/s** | Baixa |

---

## 4. Como Executar a Inferência Local via llama.cpp

### Via Linha de Comando (`llama-cli`):
```bash
llama-cli.exe \
    -m models/gguf/unrolled_modelo_qwen3loop_sft_q8_0.gguf \
    -ngl 99 \
    -p "<|im_start|>user\nCalcule o 25-ésimo termo de Fibonacci com Python.<|im_end|>\n<|im_start|>assistant\n<think>\n" \
    -n 256
```

### Servidor Local compatível com OpenAI API (`llama-server`):
```bash
llama-server.exe \
    -m models/gguf/unrolled_modelo_qwen3loop_sft_q8_0.gguf \
    -ngl 99 \
    --port 8080 \
    --host 127.0.0.1
```
