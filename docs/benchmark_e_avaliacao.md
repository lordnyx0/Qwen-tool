# Bateria de Benchmark e Avaliação de Tool Calling

Este documento detalha a metodologia de avaliação objetiva, o protocolo de testes e as métricas do benchmark de **90 itens holdout** utilizado para medir o desempenho do **Qwen3Loop**.

---

## 1. Composição da Bateria de Holdout (90 Itens)

O arquivo [`data/eval_legacy/eval_benchmark_battery_90.jsonl`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/data/eval_legacy/eval_benchmark_battery_90.jsonl) contém 90 casos de teste estritamente isolados:

| Categoria | Qtd de Casos | Descrição do Teste |
| :--- | :--- | :--- |
| **`negative_abstention`** | 30 | O modelo recebe uma lista de ferramentas, mas a pergunta do usuário é uma consulta geral (história, literatura, curiosidades) que não demanda ferramentas. O modelo **deve responder diretamente** sem emitir `<tool_call>`. |
| **`single_tool_simple`** | 30 | O usuário solicita uma ação atendida por uma ferramenta simples com 0 a 2 parâmetros (ex: calcular fatorial, obter piada, buscar filme). O modelo **deve emitir o nome correto e os argumentos exatos**. |
| **`python_math_execution`** | 30 | Questões de matemática computacional (exponenciação modular grande, primos, Fibonacci, MMC/MDC) onde o modelo **deve chamar `execute_python`** contendo o código exato que calcula a resposta correta. |

---

## 2. Critérios de Pontuação e Métricas

Cada amostra do benchmark é avaliada programaticamente por 4 critérios rigorosos:

1. **Validade Sintática do Bloco (`format_valid`):**
   - Se uma ferramenta foi chamada, o texto gerado deve conter exatamente o bloco delimitador `<tool_call>\n{JSON}\n</tool_call>`.
   - O conteúdo interno deve ser um JSON válido com chaves `"name"` e `"arguments"`.
2. **Seleção de Ferramenta (`tool_name_match`):**
   - O campo `"name"` do JSON gerado deve ser idêntico ao esperado no caso de teste (`expected_call["name"]`).
3. **Precisão dos Argumentos (`arguments_match`):**
   - Para chamadas comuns: os argumentos passados devem corresponder aos tipos e valores esperados.
   - Para `execute_python`: o código Python emitido pelo modelo é **executado em um sandbox isolado** e sua saída padrão (`stdout`) é comparada com a resposta matemática exata (`expected_execution_result`).
4. **Abstenção Negativa (`abstention_valid`):**
   - Em casos de abstenção, se o modelo emitir qualquer tag `<tool_call>`, a pontuação é zero (`0.0`).
   - Se responder apenas textualmente de forma educada e prestativa, a pontuação é total (`1.0`).

---

## 3. Script de Avaliação Automatizada

O script [`scripts/eval/benchmark_90_battery_q8.py`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/scripts/eval/benchmark_90_battery_q8.py) executa o benchmark em lote via `llama-cli`:

```bash
python scripts/eval/benchmark_90_battery_q8.py \
    --model_gguf models/gguf/unrolled_modelo_qwen3loop_sft_q8_0.gguf \
    --battery_path data/eval_legacy/eval_benchmark_battery_90.jsonl \
    --output_results data/tool_calling_q8_benchmark_results.json \
    --max_tokens 450
```

---

## 4. Resultados de Linha de Base (Baseline Base SFT vs. RL)

Resultados medidos no modelo pré-treinado antes do Fine-Tuning supervisionado de tools:

| Categoria | Acurácia Baseline (Q8_0) | Causa Principal das Falhas |
| :--- | :--- | :--- |
| **`negative_abstention`** | **100.0%** (30/30) | O modelo base já possuía excelente discernimento conversacional. |
| **`single_tool_simple`** | **56.7%** (17/30) | Em 13 casos, gerou argumentos incompletos ou texto conversacional sem emitir a tag `<tool_call>`. |
| **`python_math_execution`** | **0.0%** (0/30) | Tentava calcular mentalmente ou emitia tags de código sem estruturar como chamada da ferramenta `execute_python`. |
| **Geral Consolidado** | **52.2%** (47/90) | Necessidade do SFT especializado para converter as chamadas em Nível 2 e 3. |

### Meta pós-Tool SFT:
Atingir **> 90.0% de acurácia global** no GGUF Q8_0, com **> 85% de acertos na execução de código Python**.
