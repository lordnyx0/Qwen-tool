# 04. Checklist Sequencial e Próximos Passos para o Novo Agente

Este roteiro passo a passo deve ser seguido na ordem exata pelo novo agente para reiniciar o processo de fine-tuning supervisionado de forma limpa, auditável e bem-sucedida.

---

## 📋 Passo 1: Construção do Dataset Purificado

* [x] **Criar o script gerador:** Criar um script limpo (`scripts/pipeline/build_purified_sft_dataset.py` — criado e executado; ver `09_v2_roadmap.md`).
* [ ] **Aplicar o Catálogo Canônico (< 50 Ferramentas):**
  * Mapear chamadas para `bash`, `powershell`, `read_file`, `write_file`, `list_dir`, `python_interpreter`, `core_memory_*` e `http_request`.
* [ ] **Filtragem e Sanitização Estrita:**
  * Rejeitar qualquer comando com termos ofensivos (`mimikatz`, `dumpwcm`, `bypass`, etc.).
  * Eliminar réplicas exatas (remover multiplicações sintéticas `* 18`, `* 12`).
  * Converter qualquer chamada Hermes de string `<tool_call>` para a estrutura formal `tool_calls`.
  * Corrigir `"type": "dict"` para `"type": "object"`.
  * Eliminar amostras "zero-call" (garantir que 100% das amostras chamem ao menos uma ferramenta).
  * Exigir saídas técnicas realistas nos turnos de `role: "tool"` (nada de `"Command completed successfully with exit code 0"`).

---

## 🧪 Passo 2: Bateria de Auditoria Local (Sanidade dos Dados)

Antes de qualquer envio para a nuvem, execute a validação automatizada:

* [ ] **Validação de Sintaxe e Parse:**
  * Verificar que todos os blocos de código em Python passam por `ast.parse`.
  * Verificar que todos os valores de `arguments` em `tool_calls` são strings JSON válidas (`json.loads(arguments)` não lança exceção).
* [ ] **Validação de Tamanho (Tokenizer Oficial):**
  * Medir tokens com `AutoTokenizer.from_pretrained("checkpoints/qwen3loop_tool_sft_final")`.
  * Garantir que **nenhuma amostra ultrapasse 5.000 tokens**.
* [ ] **Verificação de ChatML:**
  * Confirmar que 100% dos turnos finais são `role: "assistant"`.
  * Confirmar que toda ferramenta em `tool_calls` está declarada em `tools`.

---

## 📤 Passo 3: Empacotamento e Publicação no Kaggle

* [ ] Copiar o arquivo final gerado para `Kaggle/kaggle_sft_data/curriculum_3000_tool_sft.jsonl`.
* [ ] Publicar a nova versão do dataset com a flag `-r tar`:
  ```bash
  python -c "
  import subprocess, os
  token = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
  env = os.environ.copy()
  env['KAGGLE_API_TOKEN'] = token
  subprocess.run(['kaggle', 'datasets', 'version', '-p', 'Kaggle/kaggle_sft_data', '-m', 'Purified SFT dataset', '-r', 'tar'], env=env)
  "
  ```
* [ ] Aguardar o status do dataset mudar para `ready`:
  ```bash
  kaggle datasets status lordnyx0/qwen3loop-tool-sft-data
  ```

---

## 🚀 Passo 4: Disparo e Monitoramento do Treinamento

* [ ] Atualizar o notebook [`Kaggle/kaggle_kernel/qwen3loop_tool_sft.ipynb`](file:///c:/Users/Nyx/Downloads/Diversos/qwen/Kaggle/kaggle_kernel/qwen3loop_tool_sft.ipynb) se houver novas flags de parâmetros.
* [ ] Enviar o kernel para execução:
  ```bash
  python -c "
  import subprocess, os
  token = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
  env = os.environ.copy()
  env['KAGGLE_API_TOKEN'] = token
  subprocess.run(['kaggle', 'kernels', 'push', '-p', 'kaggle_kernel'], env=env)
  "
  ```
* [ ] Monitorar o status de execução:
  ```bash
  kaggle kernels status lordnyx0/qwen3loop-tool-sft
  ```
* [ ] Monitorar o log de passos ao vivo:
  ```bash
  kaggle kernels status lordnyx0/qwen3loop-tool-sft
  ```
  (Nota: o `poll_logs` citado em versões antigas **não existe**.)

---

## 📥 Passo 5: Download do Checkpoint e Unrolling GGUF

Após a conclusão do kernel (`KernelWorkerStatus.COMPLETE`):

* [ ] Baixar os artefatos de saída gerados no Kaggle:
  ```bash
  kaggle kernels output lordnyx0/qwen3loop-tool-sft -p checkpoints/qwen3loop_tool_curriculum_final/
  ```
* [ ] Descompactar o arquivo `.zip` contendo os pesos finais em Safetensors.
* [ ] Executar o script de unrolling (desdobramento das 28 camadas físicas em 56 camadas lógicas) e quantização em GGUF Q8_0:
  ```bash
  python scripts/quantize/build_and_quantize_tool_sft_gguf.py
  ```
* [ ] O arquivo gerado será gravado em `models/gguf/unrolled_modelo_qwen3loop_tool_sft_q8_0.gguf`.

---

## 🎯 Passo 6: Avaliação Cega no Benchmark Holdout (90 Casos)

* [ ] Executar a bateria de avaliação contra os 90 casos de teste estritos:
  ```bash
  python scripts/eval/benchmark_tool_calling_baseline.py
  python scripts/eval/benchmark_tool_calling_q8.py
  ```
* [ ] Métricas esperadas de sucesso:
  * **Conformidade de Esquema JSON:** $\ge 98\%$
  * **Acurácia de Seleção de Ferramenta:** $\ge 92\%$
  * **Recuperação de Erro em Retrajetórias:** $\ge 85\%$
  * **Taxa de Alucinação / Ferramentas Órfãs:** $\le 2\%$
