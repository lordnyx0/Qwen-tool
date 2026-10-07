# 09. Roadmap V2 (pós-baseline do benchmark)

Parte de: meta intermediária aprovada — workspace 50→200, powershell 158→250 (~16 ciclos a 15/ciclo),
foco em bucket médio (800-2000 tokens, hoje 39.7% vs alvo 50%) e `retra:false` (retra está em 15.9%, teto 15%).

## Fila de trabalho (ordem)
1. **S5 lotes** (orquestrado, trio 10ws+5ps por ciclo): domínios inéditos, 4 turnos ws / 3-4 ps, prompts
   endurecidos (formato wrapped, escaping `\\`, ASCII). Adendo V2 (custo-zero): exigir flags Unix
   variadas (`du/df/find/grep`, estilo `--max-depth`) e paths absolutos (`/var/`, `/logs/`) nos
   args — cobre C1 (mt_001) e parte de A5 por arrasto. Gate: validador por arquivo antes de qualquer merge.
2. **Merge por lote**: backup `.bakN` → prune dead tools → append staging → retokenize no tokenizer oficial
   → re-sync `Kaggle/kaggle_sft_data/` → suite 04 Passo 2. Script canônico: `scripts/pipeline/merge_staging.py`.
3. **Validação**: `scripts/pipeline/validate_sft.py` (args parseável, call⊆declared, termina assistant, AST python,
   blocklist malware, template Qwen, PII `Elena Maria Vasquez`, placeholders).
4. **Trim de retra** se passar de 16%: priorizar `retra:false` nos lotes (já em curso).
5. **Benchmark holdout 90** (`scripts/eval/benchmark_tool_calling_baseline.py`): só após V2 fechar volume;
   correções guiadas pelos slices reprovados (esperado: workspace_code).

## Opcional / baixa prioridade
* Cauda semântica S3 (~70 `retrieve_all` com query vazia — schema válido, enriquecimento 1-a-1).
* Sampler com pesos por pilar no `train_sft_tool_qwen3loop.py` (mitiga dominância technical_api 51% sem dados novos).
* Think Nemotron (E2): reescrita só se sobrar orçamento (gravidade 4).

## Sonda in-distribution (construída, fora do treino)
* `data/eval_probe_canonical_25.jsonl` (25 itens, builder `scripts/pipeline/build_probe_canonical_25.py`,
  overlap 0 com treino): 10 L1 single, 5 L2 multi, 5 L3 python (stdout verificado por execução),
  5 abstenção. Roda no mesmo harness: `python scripts/eval/benchmark_battery_v2.py --battery_path
  data/eval_probe_canonical_25.jsonl`. Ler junto com a bateria 121: sonda alta + bateria média =
  treino OK, falta transferência; ambas baixas = problema no treino.

## Congelado em V1 (não mexer sem nova versão)
`data/cleaned/curriculum_3000_tool_sft_V1.jsonl` + `V1_MANIFEST.md`. `PURIFIED` é o working file vivo;
`Kaggle/kaggle_sft_data/curriculum_3000_tool_sft.jsonl` é o espelho publicado.
