# Distribuição do dataset v39 (N=3633, 2026-10-01)

## Como ler

Cada linha é um **lote de origem** (`source`): um conjunto de exemplos que
entrou junto, com a mesma história. Colunas: **n** = nº de exemplos;
**%** = fatia do dataset; **tools médio** = tamanho médio do catálogo
(quantas tools o modelo vê por exemplo); **calls médio** = chamadas feitas
por exemplo; **% multi-call** = exemplos com ≥2 chamadas (fluxos com
encadeamento, não tiro único).

## Os lotes, em português claro

- **purified_canonical** (1398): a massa herdada — fluxos multi-tool vindos
  da fonte ToolMind, limpos mas antigos. É o "corpo" do dataset.
- **manual_v37_small** (640): cópias com catálogo pequeno (1–4 tools),
  para o modelo treinar com poucas opções como no Pi. Quase todas single-tool.
- **manual_s5** (271): fluxos 100% multi-call (o lote mais encadeado).
- **manual_v35/P4** (200): navegação de arquivos com outputs executados de
  verdade (listar/ler/trocar após erro). Inclui pedagogia error-first.
- **manual_v37_alias** (200): as mesmas 200 do P4 com nomes de tool trocados
  (30 variantes) — ensina a abstração, não o nome.
- **manual_v31** (172): calibração workspace (inclui 5 esclarecimentos reais:
  perguntar antes de agir).
- **manual_v37/t_irr** (156): recusas com catálogo presente ("nenhuma tool
  serve") — 0 calls por construção.
- **manual_v38_beh** (140): pack cirúrgico (run-code, no-tool, políticas).
- **manual_v39_mt** (66): multi-turn real com calls após o 2º turno (novo).
- **manual_t_*** (390 no total): micro-lotes de calibração — abstinência
  (`t_abs`, sem tools), gatilhos (`t_trig`), decisões (`t_dec`), semântica,
  comandos e o t_cal (30 positivas + 30 recusas genéricas).
- **manual_v34/T-ALT** (50): navegação alternativa real (segunda leva do P4).

## Globais em claro

- **1229 tools únicas, 53,5% aparecem 1×**: diversidade enorme, profundidade
  pouca — bom para generalizar, ruim para dominar uma tool específica.
- **Catálogos 1–20 (25,9% com 1–4)**: a maioria vê muitas opções (aprende a
  escolher); um quarto vê poucas (como o Pi).
- **150 sem tools**: exemplos de abstenção pura (nada a chamar).
- **Token mediano 1152**: tamanho típico; 2 amostras passam de 5000 e o treino
  as pula em silêncio.
- **Gates**: 13 regras de validação zeradas; overlap com evals só tem
  colisões de números genéricos ("100"); schema das calls validado.

| source | n | % | tools médio | calls médio | % multi-call |
|---|---|---|---|---|---|
| purified_canonical | 1398 | 38.5% | 11.8 | 1.6 | 43% |
| manual_v37_small | 640 | 17.6% | 1.2 | 1.6 | 40% |
| manual_s5 | 271 | 7.5% | 7.2 | 3.6 | 100% |
| manual_v35 (P4) | 200 | 5.5% | 13.1 | 1.8 | 53% |
| manual_v37_alias | 200 | 5.5% | 13.1 | 1.8 | 53% |
| manual_v31 | 172 | 4.7% | 7.7 | 1.4 | 52% |
| manual_v37 (t_irr) | 156 | 4.3% | 13.5 | 0.0 | 0% |
| manual_v38_beh | 140 | 3.9% | 7.3 | 1.6 | 42% |
| manual_v39_mt | 66 | 1.8% | 8.9 | 2.9 | 91% |
| manual_t_abs | 60 | 1.7% | 0.0 | 0.0 | 0% |
| manual_t_trig | 60 | 1.7% | 12.1 | 1.0 | 0% |
| manual_t_trig2 | 60 | 1.7% | 13.4 | 1.0 | 0% |
| manual_t_cal | 60 | 1.7% | 6.1 | 0.5 | 0% |
| manual_v34 (T-ALT) | 50 | 1.4% | 12.9 | 2.3 | 62% |
| manual_t_dec | 40 | 1.1% | 13.3 | 1.0 | 0% |
| manual_t_sem | 20 | 0.6% | 12.6 | 1.0 | 0% |
| manual_t_dec2 | 20 | 0.6% | 12.4 | 1.0 | 0% |
| manual_t_cmd | 20 | 0.6% | 11.7 | 1.0 | 0% |
| **TOTAL** | **3633** | **100%** | **9.2** | **1.7** | — |
