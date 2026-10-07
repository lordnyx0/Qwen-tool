# 07. Single-Turn vs. Orfão Mid-Trajectory (Correção de Critério)

Data: sessão manual atual. Status: normativo para curadoria futura.

## 1. A distinção

* **Single-turn válido (predição, MANTER):** `user → assistant(tool_calls)` terminando aí, sem `tool` após.
  É o formato padrão Hermes/Glaive de SFT para tool-calling: o modelo aprende `P(call | user, tools)`;
  a execução ocorre na inferência, não no treino. Atende a 02 regra 4 (100% com call).
  Exemplo legítimo: `purified_0015` (`initiate_vulnerability_scan`), `purified_0096` (`schedule_smart_plug`).
* **Orfão mid-trajectory (DEFEITO, TRUNCAR):** `assistant(tool_calls)` no MEIO da trajetória cujo próximo
  turno não é `tool` (ex.: seguido de `user` ou outro `assistant`). Indica conversa truncada no meio da
  execução. Tratamento: remover o turno órfão final e manter o prefixo válido (foi o `orfao_final_truncado`
  da fase de scripts), nunca inventar observação.
* **Tool órfã (DEFEITO, REMOVER o turno):** `tool` sem `tool_calls` pendente anterior (bug do conversor
  Nemotron: `assistant` texto → `tool`). Tratamento: remover o `tool` stray, manter o resto
  (foi o `tool_stray_removido`).

## 2. Erro cometido nesta sessão (registro honesto)

O arquivo `docs/kaggle_sft_handover/exemplar_hermes_multiturn.jsonl` foi criado como gabarito de conversão
single-turn → multi-turn com `tool_response` **sintetizado** (`price_usd` inventado).
Isso é **ANTI-PADRÃO**: insere alucinação de observação no dataset.
**Determinação: NÃO aplicar esse gabarito ao lote de 227 Hermes; NÃO incluir esse arquivo no treino.
Apagar antes de qualquer `kaggle datasets version`.** Os 227 single-turn permanecem como estão.

## 3. Por que manter single-turn não quebra o modelo

Template Qwen renderiza `tool_calls` como tags `<tool_call>` nos dois casos; o vocabulário de saída é o
mesmo, muda só o condicionamento (`P(call|user)` vs `P(call|histórico com observations)`).
Hermes, Glaive e ToolACE misturam os dois regimes por padrão. Riscos residuais reais, todos gerenciáveis:
1. Viés de disparo único — excesso de single-turn ensina a emitir todas as calls de uma vez e nunca iterar.
   Controle: pilar multi-turn é majoritário + retrajectories em 13.7% ensinam iteração.
2. Falta de grounding — single-turn não ensina reação a observation; coberto pelo pilar retra.
3. Risco MAIOR seria o oposto: converter com observations fabricadas (o gabarito acima) ensinaria o modelo
   a inventar resultados de ferramenta. Por isso a determinação do item 2.
