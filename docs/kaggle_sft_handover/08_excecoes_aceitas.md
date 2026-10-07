# 08. Exceções Aceitas (documentar, não corrigir)

Escala de gravidade: **1 = altamente problemático** (corrompe treino / quebra inferência),
2 = alto, 3 = moderado, 4 = baixo, 5 = cosmético.
Custo em ciclos manuais/orquestrados (1 ciclo ≈ 1 trio de 15 + validação + re-sync).

## E1. `technical_api` com ~1000 tools vs 02 `<50`
* **Gravidade: 2.** Desvio grande da letra da 02, risco real moderado-baixo: cada amostra carrega o
  próprio schema no contexto, então o modelo aprende *seguir-schema* (habilidade que generaliza),
  não decorar nomes. A esparsidade (maioria 1-shot) limita aprendizado de semântica por ferramenta,
  mas esse nunca foi o objetivo do pilar — o sinal dele é tipagem estrita (ver 06: matriz 16k).
* **Custo de correção: ~100 ciclos** (reescrever ~1000 tools+args para 4 genéricas), **e destrutivo**:
  apagaria o único sinal de tipagem estrita do dataset.
* **Decisão: ACEITAR.** O `<50` vale para terminal/workspace/memory (cumprido); técnica diversa é
  exceção fundamentada na 06.

## E2. `<think>` Nemotron verboso e genérico (~394 amostras)
* **Gravidade: 4.** Formato válido (template extrai `reasoning_content`), conteúdo fraco
  ("We are in a fresh terminal...", "Let's first check..."). Não corrompe; sub-otimiza o reasoning
  e reforça openers repetitivos.
* **Custo de correção: ~40 ciclos** (reescrever 400 thinks com deliberação estratégica real).
* **Decisão: ACEITAR por enquanto.** ROI baixo vs S5; candidato a melhoria futura se sobrar orçamento.
  Mitigação parcial já aplicada: boilerplate do `user` removido (1 Edit, 0 matches restantes).

## E3. 9 amostras com pilar/tool fora do canônico (7 `search_engine_query` + 2 airline sob `workspace_code`)
* **Gravidade: 3.** 0.5% do dataset (9/1938); pilar errado distorce a distribuição de pilares e a tool
  não está nas 15 canônicas. Args são válidos e observations substantivas — o defeito é taxonômico.
* **Custo de correção: ~1 ciclo.** Repilarar para `technical_api` mantendo a tool como exceção
  (mapear `search_engine_query→http_request` exigiria fabricar URLs — proibido pela 07).
* **Decisão: EXECUTADA.** Repilarados via script mecânico (backup `.bak2` prévio): 7 search
  (`purified_0314/0441/0514/0994/1242/1272/1562`) + 2 airline (`purified_0761/1873`) → `technical_api`.
  Workspace restante: 20 amostras, 100% `read_file`/`list_dir` genuínos (verificado por `Grep`).
  Tools `search_engine_query`/airline mantidas como exceção (mapear fabricaria URLs).

## Nota sobre os "40 ciclos"
Ver conta em separado com o usuário; resumo: 40 é o piso otimista (trio perfeito, zero retrabalho).
Observado nesta sessão: 2 de 3 arquivos do trio precisaram de correção — throughput real ~10-12
amostras boas/ciclo, ou seja, **45-55 ciclos para as ~600 restantes**. Alternativa: meta intermediária
workspace 29→200 + powershell 143→250 (~280 amostras, **19-25 ciclos**), que já entrega 8x de workspace
e fecha o bucket médio perto dos 50%.

## E9. Outputs vazios em trajetórias bash (staging `nemotron_easy_fileops`, 4797)
* Regra `generica` acusa ~85k tool-outputs vazios/curtos. Em terminal,
  stdout vazio é legítimo (`mkdir`, `cd`, `touch` não emitem nada) —
  ausência de output NÃO é ausência de evidência.
* **Decisão: ACEITAR com escopo.** `generica` vale para narrow tools
  (read_file vazio = suspeito); para `bash`, vazio é normal. Exceção
  restrita a este staging; se fundir, reavaliar por amostra.
