# 10. Plano de ação — dataset V2 (mecânico primeiro)

Origem: diagnóstico de causas (A1–A6, B, C) + plano de lotes aprovado.
Meta: `data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl` 1983 → ~2346 amostras (+18%),
retra 15.9% → ~13.4%, zero edições em amostras existentes, 59 descartes mecânicos.
Ordem: mecânico e barato primeiro; manual pesado por último. S5 corre em paralelo o tempo todo.

Regras vigentes em todas as etapas (ver AGENTS.md na raiz):
CWD = raiz do repo; single-writer no PURIFIED; staging entra só via
`scripts/pipeline/merge_staging.py`; backup `.bakN` antes de cada merge;
`scripts/pipeline/validate_sft.py` verde após cada lote; promessa de lote ≠ entrega.

## Etapa 1 — Dedup mecânico das queries vazias (10 min, script)

Descarta os 59 excedentes dos 16 grupos de duplicata exata em
`archival_memory_search` com query vazia (levantamento em script de escopo já executado).
Os `retrieve_all` únicos permanecem (exceção vigente).

1. Escrever o script de dedup em `scripts/pipeline/` (regra: match exato em
   `(user[:80], arguments)`; resto intocado).
2. Rodar em dry-run, conferir os 59 ids listados.
3. Backup `.bakN` do PURIFIED → aplicar → rodar `scripts/pipeline/validate_sft.py`.
4. Gate: N = 1924, suite 0 fails, diff mostra só remoções.

## Etapa 2 — Harness V2-eval (código, sem dado novo)

Aposenta ~14 FAILs como classe de medida, para o placar V2 comparar limpo contra os 70.2%:

1. Em `scripts/eval/benchmark_battery_v2.py` (ou runner novo na mesma pasta): para
   `expected_tool == execute_python`, comparar **stdout executado** em vez de string de código
   (reaproveitar a função de sandbox que já existe no arquivo).
2. Normalização leve de args: strip de pontuação final e alias de unidade (`C`→`celsius`).
3. Gate: re-rodar a bateria EN + sonda; nenhum PASS anterior pode virar FAIL
   (só FAILs-artefato podem virar PASS; documentar quais viraram no CHANGELOG do handover).

## Etapa 3 — Gate de execução do T-TRIG (script, antes de authorar)

Validador mecânico das 60 futuras amostras trigger: extrai o snippet `code` de
`expected_args`, executa em sandbox com timeout, compara com `ground_truth_stdout`.

1. Script novo em `scripts/pipeline/` (entra como gate oficial do lote).
2. Gate: roda verde sobre os 30 itens L3 da bateria EN como positivos (são referências
   corretas — o gate valida a *amostra*, não o modelo) + 4 negativos sintéticos
   (divisão por zero, stdout trocado, heredoc, code vazio) todos capturados.
   Script canônico: `scripts/pipeline/validate_ttrig.py --selftest`.

## Etapa 4 — Adendo custo-zero na spec do S5 (doc, 5 min)

Editar a fila do `09_v2_roadmap.md`: nos trios ws/ps, exigir flags Unix variadas
(`du/df/find/grep`, inclusive `--max-depth`-style) e paths absolutos nos args.
Custo marginal zero na redação, cobre C1 e parte de A5 por arrasto.

## Etapa 5 — T-SEM: 20 amostras (manual leve)

10 pares lock/unlock contrastivos + 10 com paths absolutos (`/var/`, `/logs/`).
Formato single-turn curto, `retra:false`, ferramentas já conhecidas.
Staging separado em `data/staging/` (um por lote) → validador por arquivo → merge.

## Etapa 6 — T-ABS: 60 amostras (manual médio)

Abstenção pura, `retra:false`, **zero `tool_call`** na resposta: prompt pede tool
inexistente/fora de escopo → texto curto declinando com motivo.
Variar domínios (terminal, finanças, smart home, memória, web) para não criar novo prior.
Gate extra além do validador: grep `tool_call` no staging tem que retornar 0 hits.

## Etapa 7 — T-TRIG: 60 amostras (manual médio + gate)

Pergunta quantitativa direta → `execute_python` com snippet `print` (estilo L3 da bateria,
não heredoc). Cada amostra passa pela Etapa 3: divergiu → corrijo na hora ou descarto.
Se o descarte passar de 10/60, pausar e revisar o template de redação antes de continuar.

## Etapa 8 — T-DEC: 40 amostras (manual pesado, por último)

Financeiras + pares contrastivos: mesmo-quase prompt, tools vizinhas
(`get_stock_quote` × `get_company_financials`, `calculate_crypto_swap_rate` ×
`verify_crypto_gas_price`, `simulate_loan_amortization` × `calculate_compound_interest`),
uma só certa, valores plausíveis. É o lote mais caro — fazer com calma, 10 pares por sessão.
Gate: para cada par, trocar as tools de lugar não pode trocar o gabarito
(anti-vazamento pelo exemplo).

## Etapa 9 — Merge final e contagem

1. Um merge por lote (backup + prune + append + retokenize + re-sync `Kaggle/kaggle_sft_data/`,
   via `scripts/pipeline/merge_staging.py`).
2. `scripts/pipeline/validate_sft.py` + `scripts/pipeline/validate_docs.py` verdes.
3. Conferir a conta: N ≈ 2346, retra ≈ 13.4%, buckets e pilares contra
   `docs/kaggle_sft_handover/V1_MANIFEST.md`.
4. Registrar tudo no CHANGELOG do handover. Só então liberar re-treino.

## Fora deste plano (V3, não começar)

Calibração anti-alucinação, auditoria runtime dos 541 códigos antigos,
enriquecimento das ~250 queries vazias restantes.

---

## Checklist

- [x] Etapa 1: script dedup escrito; dry-run lista 59 ids; backup feito; N=1924; validador verde
- [x] Etapa 2: harness compara código por execução; re-run EN sem regressão; viradas PASS documentadas
  (executado: 0 PASS→FAIL; bateria 85→86, sonda 4→9; só artefatos viraram)
- [x] Etapa 3: gate T-TRIG pronto — 30/30 referências L3 OK + 4/4 negativos capturados
  (`scripts/pipeline/validate_ttrig.py --selftest`; gate valida a amostra, não o modelo)
- [x] Etapa 4: `09_v2_roadmap.md` com adendo de flags/paths; `validate_docs.py` verde
- [x] Etapa 5: 20 T-SEM em staging; validador por arquivo verde; merge feito (N=1944)
- [x] Etapa 6: 60 T-ABS em staging; grep `tool_call` = 0 hits; merge feito (N=2004;
  fix no merge: abstencao legitima permite tools vazias quando zero calls)
- [x] Etapa 7: 60 T-TRIG (descarte ≤ 10); gate de execução 60/60; merge feito
  (executado: gate pegou 2 deslizes de aritmética minha, corrigidos; 0 descartes; N=2064)
- [x] Etapa 8: 40 T-DEC (20 pares); anti-vazamento checado; merge feito
  (executado: posição da tool certa 20/20; schemas decoy normalizados; N=2104)
- [ ] Etapa 9: N ≈ 2346; retra ≈ 13.4%; validadores verdes; CHANGELOG atualizado; re-treino liberado
  (parcial em 2026-09-27: T-lotes prontos, N=2104, retra 14.6%; faltam os 242 do S5)
