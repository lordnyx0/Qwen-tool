# V1 Manifest — congelado para treino

* **Payload:** `Kaggle/kaggle_sft_data/curriculum_3000_tool_sft.jsonl` (8.671.336 bytes) + espelho
  `data/cleaned/curriculum_3000_tool_sft_V1.jsonl`. Stats em `data/cleaned/curriculum_V1_stats.json`.
* **N = 1983.** Pilares: technical_api 1012, terminal_linux 394, memory_retrieval 369,
  windows_powershell 158, workspace_code 50. Retra 315 (15.9%). Buckets: curto 829 (41.8%),
  médio 788 (39.7%), longo 366 (18.5%). Token max 4228, mediana ~943.
* **Gates 04 Passo 2 (todos zero):** args JSON parseável, call⊆declared (declared==called==1210),
  termina em assistant, `ast.parse` nos blocos python, 0 malicioso (blocklist), template Qwen 0/1938→0/1983.
* **Exceções vigentes (08):** E1 técnica diversa, E2 think verboso, E3 resolvida (repilarado).
  `archival_memory_search.query` opcional (retrieve_all legítimo).

## Por que treinar já
1. Todos os vícios *corruptivos* acabaram (malware, fake obs, JSON quebrado, esquizofrenia de formato,
   declarações mortas, boilerplate). O que resta é *quantitativo* (balanço de pilares/buckets).
2. Sem baseline, a meta V2 é chute: o benchmark 90 holdout (metas ≥98% schema, ≥92% seleção,
   ≥85% retra, ≤2% alucinação) dirá exatamente onde dói — provavelmente workspace (2.5% do treino).
3. Custo marginal ~zero (Kaggle T4 gratuito); V1 treina enquanto S5 continua.
4. Mitigação barata do desbalanceamento sem dados novos: sampler com pesos por pilar no
   `train_sft_tool_qwen3loop.py` (1 edição mecânica futura).

## Escopo V2 (pós-benchmark)
* S5: workspace 50→200, powershell 158→250 (~16 ciclos); priorizar bucket médio e `retra:false`
  (retra está em 15.9%, acima do teto).
* Correções guiadas pelos slices reprovados no holdout (esperado: workspace_code, seleção powershell).
* Cauda semântica S3 opcional (~70 retrieve_all com query).

## Registro de treino V1
* Dataset `lordnyx0/qwen3loop-tool-sft-data`: **ready**. Kernel `lordnyx0/qwen3loop-tool-sft` **v21 pushed**,
  status inicial QUEUED (fila T4). Comando: DDP 2 proc, batch 1, grad_accum 8 (efetivo 16), 1 epoch (~118
  optimizer steps), lr 1.5e-5, fp32, eval 25 / save 35. ETA 2-4h após sair da fila.
* Monitorar: `kaggle kernels status lordnyx0/qwen3loop-tool-sft` (nota: o `poll_logs` citado em versões antigas **não existe**). Ao COMPLETE: download output →
  `scripts/quantize/build_and_quantize_tool_sft_gguf.py` → benchmarks (metas 04).
