# Currículo e Datasets de Tool Calling: Qwen3Loop

Este documento descreve a taxonomia de dados, fontes de extração, metodologia de balanceamento e formatação de tokens do ecossistema de dados do **Qwen3Loop**.

---

## 1. Visão Geral da Taxonomia por Níveis

Para treinar um modelo de 0.6B com proficiência agêntica sem saturar sua capacidade de representação, os dados são organizados em uma estrutura de currículo progressivo de três níveis:

```
[Nível 1: Foundation & Abstention] ──> [Nível 2: Discrimination & Schemas] ──> [Nível 3: Algorithmic & Python]
      - 500 Single Tools                     - 1.071 Multi-Tools Concorrentes          - 1.000 Problemas Matemáticos
      - 500 Abstenções Negativas             - Schemas Aninhados & Arrays              - Código Python Exato
```

---

## 2. Detalhamento dos Componentes

### 2.1 Nível 1: Foundation & Abstenção Conversacional
- **Objetivo:** Estabelecer o comportamento atômico de invocar ferramentas quando necessário, e **abster-se formalmente** quando o usuário faz perguntas gerais ou pede tarefas não suportadas pelas ferramentas disponíveis.
- **Fontes de Dados:**
  - *Glaive Function Calling v2*: Amostras limpas com apenas 1 ferramenta no prompt e no máximo 2 parâmetros primitivos (strings, inteiros, booleanos).
  - *Abstenções Glaive & Sintéticas*: Consultas cotidianas (história, literatura, curiosidades) apresentadas com um catálogo de ferramentas irrelevantes para forçar o modelo a responder textualmente sem emitir `<tool_call>`.
- **Volume:** 1.000 amostras (500 positivas + 500 negativas).

### 2.2 Nível 2: Discriminação e Schemas Estruturados
- **Objetivo:** Treinar o modelo para escolher a ferramenta exata entre 2 a 6 ferramentas candidatas no catálogo, e preencher argumentos com tipos de dados compostos (objetos JSON aninhados, arrays, filtros de data).
- **Fonte de Dados:**
  - *NousResearch/hermes-function-calling-v1*: Dados de alta fidelidade extraídos do Hermes 2 Pro.
- **Volume:** 1.071 amostras.

### 2.3 Nível 3: Execução de Código Python (`execute_python`)
- **Objetivo:** Resolver problemas matemáticos exatos onde LLMs compactos alucinam (exponenciação modular grande, fatoriais, primos, MMC/MDC, termos avançados de Fibonacci) delegando o cálculo a um interpretador Python em tempo de execução.
- **Schema da Ferramenta:**
  ```json
  {
    "type": "function",
    "function": {
      "name": "execute_python",
      "description": "Executa código Python em ambiente isolado e retorna o valor de saída padrão (stdout) ou resultado.",
      "parameters": {
        "type": "object",
        "properties": {
          "code": {
            "type": "string",
            "description": "Código Python válido a ser executado."
          }
        },
        "required": ["code"]
      }
    }
  }
  ```
- **Volume:** 1.000 amostras (619 da base inicial + 381 geradas sinteticamente e verificadas matematicamente via Python).

---

## 3. Dataset Holdout de Avaliação (`eval_benchmark_battery.jsonl`)

Para garantir que a avaliação seja rigorosamente cega e confiável:
- **Tamanho:** 90 itens (30 Nível 1 Single-Tool, 30 Nível 1 Abstenção, 30 Nível 3 Python Math).
- **Proteção Contra Vazamento (Zero Leakage):** Durante a geração do dataset de treino ([`scripts/build_archive/build_clean_curated_sft_dataset.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scripts/build_archive/build_clean_curated_sft_dataset.py)), todos os 90 prompts do benchmark são normalizados em texto e indexados. Qualquer amostra de treino com sobreposição textual ou semântica é **estritamente descartada**, garantindo 0 amostras vazadas no conjunto final de 3.071 exemplos.

---

## 4. Formato de Tokenização e ChatML

Todas as amostras são estruturadas seguindo a especificação Qwen ChatML:

```xml
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
{pergunta_do_usuario}<|im_end|>
<|im_start|>assistant
<think>
{raciocinio_conciso}
</think>

<tool_call>
{"name": "{nome_da_tool}", "arguments": {...}}
</tool_call><|im_end|>
```

### Regra de Máscara de Perda:
No treinamento supervisionado, todos os tokens até `<|im_start|>assistant\n<think>\n` recebem o rótulo `labels = -100`. Desta forma, o modelo é penalizado exclusivamente pela produção do raciocínio analítico e da chamada estruturada.
