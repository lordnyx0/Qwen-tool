# 13 — V35: plano de correção pós-auditoria externa

Data: 2026-09-30. Status: proposta (aguarda parecer do revisor + decisões).
Base: PURIFIED N=2720 (`data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl`).
Restrição dura: **v34 está em treino (sessão 354117130) — nada deste plano toca o v34.**
Tudo aqui mira v35. Single-writer no PURIFIED; staging via
`scripts/pipeline/merge_staging.py`; gates `validate_sft.py` (0 em tudo),
`check_eval_overlap.py` (0 HARD) e `validate_docs.py` (0 quebras) após cada lote.

## 0. Contexto aceito da auditoria (+ réplica verificada)

Réplica confirmou: repeats 63=37+14+12 exato (23 com 2º erro); s5_b04a_0003;
denominador 1935; 908/1686; split 123 evitáveis vs 242 imprevisíveis.
Correções aceitas da réplica, verificadas: **C1 cobre 20/63** (não 63);
**paginação-repeat é classe própria** (~36 s5: 20 read + 16 ps — cursor
implícito, nocivo; meu "menos nocivo" estava errado); **Q3 está contido em Q2**
(N=2269 nos dois casos; "~2190" era contagem dupla, decisão Q3 some);
`required` omite (100%-rule é anti-conservadora em defs finas); inferir só de
calls com sucesso; máscara de schema-violation por padrão pós-C2;
dose-resposta em tranches de 50 avaliadas cumulativamente (+50/+100/+150/+200);
P0/P5 invertido (mini-harness de conversas reais do Pi ANTES do sweep);
seeds 1@0.0 + 5@0.3/0.8; limiar numérico + comparação pareada (2pts < ruído
em n=60); attribution expectations (v35 em bundle só isola causa via ablation).
Q2 custa 41.0% dos tokens (verificado) e terminal 428→34; first-think
restabelece o enunciado parcialmente (opção de recuperação anotada, não default).
Schedule/LR: N menor muda steps/época — comparar v34×v35 por steps totais.

## 1. P0 — Mini-harness de navegação, depois sweep (local, sem Kaggle)

Ordem invertida (aceito): primeiro P5 (`data/eval_nav_p5.jsonl`, harness local
FORA das baterias congeladas) — 24 tarefas montadas das conversas reais que
falharam no Pi (gate 6 verbatim + loops `list_dir`/`tasklist` + 15 near-variants
de mesmo modo de falha), toolset espelhando o Pi (sem `list_dir`), metade com
erro prévio embutido (estilo multiturn: observation → correção esperada).
**P5 entra no `check_eval_overlap.py` como eval (HARD gate) ANTES de qualquer
tranche.** Viés declarado: montado de falhas, superestima taxa absoluta — vale
para comparar versões entre si, nunca como número absoluto. Baseline
obrigatório: v33_ep2 (agora) e v34 (quando terminar) no P5 antes do v35.
Depois o sweep no P5 (+ dev-hard secundário): temps {0.0, 0.3, 0.8} × seeds
{1 @ 0.0, 5 @ 0.3 e 0.8} × `repeat_penalty` {1.0, 1.08}; unidade = item (taxa
por item agregada nas seeds; seeds no mesmo item não são independentes).
Métricas: JSON válido, primeira tool correta, fidelidade + taxa de loop.
Regra pré-registrada: **decoding-domina se falha@0.0 ≤ 25% E falha@0.8 ≥ 60%**
(pareada por item); **dado-domina se falha@0.0 ≥ 50%**; senão misto → v35
completo. Paralelo no Pi: registrar o sampling real enviado (logs do servidor)
e repetir o gate em temp baixa.

## 2. P1 — Quarentena mecânica (sem juízo de conteúdo, reversível)

Regras determinísticas, saída para staging de quarentena (não delete):
- Q1 (57): `tools==[]` AND menção a tool específica não declarada
  (regex de "only X is/are declared", "my only tool"). Remove do treino v35.
- Q2 (394): user-turn == exatamente o boilerplate "Solve the terminal task
  below step by step with bash." (tarefa perdida, irrecuperável; ensina
  hallucination-of-intent). Quarentena default; rollback preservado.
  (Q3 removida: os 79 "Are you sure" + 394 `root@` estão 100% contidos em Q2 —
  N=2269 nos dois casos; "~2190" era contagem dupla. Verificado.)
- QP (paginação-repeat, **43 verificados**: 36 s5 + 7 purified; 0 manual_v34;
  os "48" não reproduzem em nenhuma definição — exata, normalizada ou
  any-pair; diferença de 5 uuids pedida ao revisor): mesma call, mesmos args,
  conteúdos DIFERENTES (cursor implícito — nocivo como loop).
  Quarentena default (1.6% do corpus); purified amostrados 1 a 1 (parte pode
  ser legítima: arquivo mudou entre calls).
Impacto: Q1=57, Q2=394, QP=43, uniões com 1 overlap (Q2&QP) → **N=2227**.
Nenhuma amostra é apagada;
quarentena fica em `data/staging/quarantine/` com o motivo por uuid.
Custo Q2 anotado: 41.0% dos tokens, terminal 428→34; first-think restabelece
o enunciado parcialmente (recuperação a partir do think NÃO é default —
fabricaria user-turns; só com ordem explícita). Schedule: N menor muda
steps/época — v34×v35 comparados por steps totais, não por épocas.

## 3. P2 — Reescrita manual (autoria, outputs reais executados)

- R1 (14): T-ALT-A/A2 violam doc-12 (retry idêntico). Reescrever call_2 como
  tentativa VARIADA (erro distinto, outputs E* já executados em arquivo) +
  think de troca. Mantém a lição loop-break sem modelar o proibido.
- R2 (57): abstenções Q1 reescritas na forma genérica ("não tenho tool para
  isso", sem citar ausentes). **Pendente ordem explícita de autoria.**
- R3 (12): mkdir-literal já fecha 12 (T-ALT+T-ALT2); sem ação salvo sinal do v34.

## 4. P3 — Código (treino + inferência de schema)

- C1 (loss-mask, `scripts/train/train_sft_tool_qwen3loop.py`): mascarar (-100)
  turnos do assistente cuja call repete (nome+args) call anterior do mesmo
  sample que foi seguida de erro. **Escopo corrigido: cobre 20/63, não 63**
  (43 repetem calls com sucesso — classe QP, tratada em P1, não no mask).
  Teste: py_compile + smoke `--max_steps` curto + diff de labels em fixture.
- C1b (máscara pós-C2, default): após inferir schemas, calls que violam o
  schema visível (args ∉ properties) são máscara mecânica — sem ela, C2
  converte o erro-first em "evitável" não-defensável (interação aceita).
- C2 (schema-inference, script novo em `scripts/pipeline/`): para defs vazias
  com calls com args, inferir `type` por valor (string/number/boolean/
  object/array) **só de calls sem erro** (contaminação aceita; ~142 defs sem
  nenhum sucesso ficam sem inferência); **`required` sempre omitido**
  (100%-rule é anti-conservadora em defs finas — 714/1001 single-call).
  Regenerar + revalidar gates + auditoria amostral (20 defs).

## 5. P4 — Dados de navegação (adição, com porta de entrada externa)

Alvo: +200 em 4 tranches de 50 (cada uma com gates), avaliadas cumulativamente:
treina-se SEMPRE nos pontos combinados (N_Q+50, depois N_Q+100 totais, ...),
nunca em tranches isoladas — não há "de graça", cada ponto é um treino.
Escala além de +200 só se a curva não saturar. (NÃO 1–3k: monotonia de
template num 0.6B.) Cobertura: `ls`→escolher→`cat/head`→resumir; README/package;
diretório grande truncado; vazio; inexistente; cwd explícito — tudo com
execução real em sandbox + fixtures neutras, formato `s5_ws_0001`, EN,
overlap 0 HARD. Oferta do revisor (gerador sandbox) aceita SE passar pelos
mesmos gates + auditoria amostral; alternativa é lote manual no mesmo molde
T-ALT (precedente: 50 amostras com outputs reais).

## 6. P5 — Harness fixo (resumo operacional; definição na §1)

`data/eval_nav_p5.jsonl`, 24 tarefas, fora das baterias congeladas.
Governança de evals intacta. Uso: leitura do sweep (P0), baselines v33_ep2/v34,
comparação v34 vs v35 e ablations. Overlap HARD contra o treino via
`check_eval_overlap.py`.

## 7. Ablation (condicional ao P0)

Se temp não explicar: A = v35 completo vs B = v35 com loss mascarado nas
calls evitáveis (EISDIR + schema-violation detectável por padrão). Compara
harness P5 (primário) + dev-hard L4 + bateria, com comparação pareada por item
e piso de ruído (2pts em n=60 ≈ 1 item — abaixo do ruído; sem "vitória" sem
significância). Custo honesto: 2× SFT (~8h T4). Com C1+C1b aplicados por
padrão, o ablation vira confirmatório — vale as 8h só se o P0 der negativo.

## 8. Versão v35

PURIFIED − Q1/Q2/QP + R1/R2 + C1/C1b/C2 + P4(tranches) → revalidar 3 gates → sync
`Kaggle/kaggle_sft_data/` → versionar (`-r tar`) → push kernel v35 → run.
Fora de escopo: corte de `technical_api` (sem evidência de starvation),
evals congelados, track RL (estacionado), `.bak`/caches.

## 9. Perguntas ao revisor (rev. 2 — parte respondida na réplica)

1. Quarentena Q2 (394, 41% dos tokens): mantém default quarentenar? A leitura
   alternativa (bash-fluency compensa) perdeu força com o custo medido; fica
   a decisão + nota de schedule (steps/época mudam).
2. ~~Regra de `required`~~ Respondida e aceita: omitir sempre. (Virada para o plano.)
3. ~~s5_b04a_0003 (i)/(ii)~~ Virou classe QP (~36): default quarentena; offset/limit
   só se quiser ensinar paginação de verdade. (Virada para o plano.)
4. Scripts: auditar os nossos aqui; fidelidade de saída como critério único
   (aceito — vale pros dois lados, com gate de amostragem).
5. ~~Alvo P4~~ Dose-resposta em tranches de 50 (aceito); começar em 200 direto
   vs cumulativo: fomos de cumulativo (+50…+200 de graça). Objeção?
6. ~~Ablation~~ Condicional ao P0 negativo + pareada + confirmatória (aceito).
   Nova: gate de amostragem dos 11 purified QP (legítimos?) — quem amostra?
