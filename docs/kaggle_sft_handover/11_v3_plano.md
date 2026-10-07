# 11. Plano V3 — correções do V2 (escopo fechado, sem espelhar eval)

Lições do V2 (87/121, +1): abstenção e decoys generalizaram (vetores validados);
trigger melhorou só em instâncias espelhadas e piorou na sonda limpa (contaminação);
60 "cale-se" deslocaram o limiar (8 silêncios indevidos). V3 = refazer esses dois
vetores com metodologia limpa + S5 pendente. Sonda/bateria continuam congeladas:
nenhuma amostra V3 pode compartilhar instância (números, entidades, statements)
com `eval_benchmark_battery_v2_en`, sonda 25 ou multiturn 12 — gate mecânico abaixo.

## Lotes (402 novas; N 2104 → ~2506; retra 308/2506 = 12.3%)

### T-TRIG2: 60 triggers com instâncias frescas (manual médio)
Mesmos conceitos do L3 + tipos novos (string ops, date math, conversão via código),
mas NÚMEROS e entidades inéditos — zero sobreposição com qualquer item de eval.
40 mesmos-conceitos-outros-números + 20 tipos novos. Gate: `validate_ttrig.py`
(exec 60/60) + gate novo de overlap (abaixo). Orçamento de descarte: 10.

### T-CAL: 60 calibração agir/calar (manual pesado, o lote mais importante)
30 pares contrastivos mínimos: mesmo domínio e mesmo toolset declarado, um prompt
coberto pela tool (→ chama) e um quase-idêntico fora de cobertura (→ declina com
motivo). Ex.: "converta 350 USD" (chama) vs "aprove a transferência de 350 USD"
 declina — mesma frase-base, fronteira na cobertura, não no tema.
Ensina a FRONTEIRA em vez do silêncio. Gate: 30/30 pares com gabaritos opostos
corretos + grep `tool_call` = 0 hits só nas metades decline.

### T-DEC2: 20 pares financeiros, entidades frescas (manual leve)
Extensão do vetor validado: mesmos 8 tools, tickers/valores/números novos
(nada de AAPL/PETR4/42km — trocar tudo). Posição da certa 10/10. Gate: validador.

### T-CMD: 20 cmdlet grounding PowerShell (manual leve)
Só cmdlets reais (inventário dos 33 do treino + comuns: `Get-Service`,
`Get-Process`, `Get-ChildItem`, `Get-EventLog`, `Stop-Service -Force`...),
completar/fixar comando sob observation de erro. Mira a classe `probe_003`
(invenção `Get-X`). Gate: todo cmdlet do lote existe na lista-base (grep mecânico).

### S5: 242 (em curso, fora deste loteamento)
ws 150 + ps 92 + adendo de flags/paths já vigente no `09`.

## Gate novo: overlap treino↔eval (mecânico, antes de qualquer merge)

Script em `scripts/pipeline/`: para cada amostra de staging, extrai números,
literais quoted e queries; falha se qualquer um aparecer em
`eval_benchmark_battery_v2_en.jsonl`, `eval_probe_canonical_25.jsonl` ou
`eval_multiturn_v1.jsonl` (exceção: scaffolding genérico como `print(`, `import math`).
Retroativo: rodar também sobre T-TRIG/T-DEC/T-SEM (documentar o que acusa, sem reescrever
o passado — serve de linha de base da contaminação V2).

## Barra de ship do modelo V3 (combinado antes do treino)

1. Strict bateria EN ≥ 75% (hoje 71.9% v2eval: 87/121).
2. Zero regressões L1 (30/30 de volta) e ≤2 silêncios indevidos novos.
3. Sonda python ≥ 3/5 (hoje 2/5) — leitura limpa do trigger.
4. Abstenção L4 ≥ 14/20 (hoje 14/20 — manter com menos silêncio indevido).
5. Multiturn ≥ 10/12. Se qualquer barra falhar, V3 não shippa (vira diagnóstico, como V2).

## Ordem e checklist

1. Gate de overlap (script) — antes de authorar qualquer coisa.
2. T-TRIG2 (60) → gate exec + overlap → staging.
3. T-CAL (60) → gate pares + grep → staging.
4. T-DEC2 (20) + T-CMD (20) → validador → staging.
5. S5 em paralelo o tempo todo.
6. Merges por lote (backup + validador) → contagem (N≈2506, retra≈12.3%) →
   re-sync Kaggle → treino da base → eval nas 3 baterias contra as barras.

- [x] Gate de overlap escrito; roda limpo sobre T-TRIG2/DEC2/CAL/CMD; baseline V2 documentada
  (baseline: t_trig 39 overlaps = contaminação confirmada; t_sem entidades + t_abs fracos)
- [x] T-TRIG2 60/60 exec + 0 overlaps; merge feito (N=2164; gate do autor pegou
  3 deslizes de aritmética, corrigidos via stdout executado)
- [x] T-CAL 30/30 pares opostos corretos; merge feito (N=2224; reconstruído via builder
  após redação rápida quebrar JSON; schemas normalizados; 0 HARD)
- [x] T-DEC2 20 + T-CMD 20 (cmdlets na lista-base); merge feito (N=2264; 0 HARD;
  builders programáticos, sem JSON manual)
- [x] S5 242 integrados; N≈2506; retra≈12.3%; validadores verdes
  (executado: b06-b21 = 150ws+92ps; −8 dups exatas do canônico → **N=2498**;
  **retra real 22.0%** — S5 é correction-arc por desenho, logo conta como retra;
  teto de 15% do `09` pressupunha retra=loop-de-erro: premissa superada, aceitar e
  monitorar cautela excessiva no eval; payload re-syncado)
- [ ] Modelo V3 treinado da base; 5 barras de ship verificadas
