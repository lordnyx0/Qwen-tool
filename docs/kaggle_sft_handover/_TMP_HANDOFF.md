# TMP — Handoff para sessão sucessora (DELETAR após uso)

Data: 2026-09-30. Contexto: sessão longa, v35 cancelado pelo usuário,
v36 por montar. Sem validação aqui propositalmente em paths (para não
quebrar o validate_docs); confira existências antes de usar.

## 1. Estado em 30 segundos

- Treino v35 CANCELADO pelo usuário (sessão 354431631, parou aos ~35 steps;
  nada aproveitável). Motivo: buscar melhor dataset antes de relançar.
  Usuário relança sozinho; meu trabalho é deixar o v36 pronto.
- Dataset vivo: data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl, N=2484.
  É o v35 + fixes pós-v35 (R1/R2, C2-refill, 2 required). Kaggle ainda tem
  o snapshot v35 (6.312.898 bytes); payload local Kaggle/kaggle_sft_data/ está
  DESATUALIZADO (train sem regex apertada). Re-sync obrigatório antes de versionar.
- Kernel Kaggle: v35 pushado (notebook inalterado desde v33; todos os args
  novos têm default, então v36 provavelmente não precisa mudar o notebook).
- Modelo em avaliação: GGUF Q8 da época 2 do v34
  (models/gguf/unrolled_modelo_qwen3loop_tool_sft_v34_ep2_q8_0.gguf):
  4/5 barras (só L4 12/20 < 14/20). Gate Pi real: 1.5/6 a temp 0.8, 3/6 a temp 0.0.

## 2. Fila imediata (ordem)

1. **Reescrita das 266+28 descrições** (achado novo do crítico, NÃO executado):
   read_file citando list_dir não declarada (171 P4 + 31 v31 + 31 v34 + 5
   purified) e list_dir citando read_file (28). Regra mecânica: se bash nos
   tools → frase aponta `bash (ls)`; se powershell → `Get-ChildItem`;
   senão, frase genérica de shell. Ver contagens em scratch crit12.py.
2. **R2-15b**: t_cal_15b ("only toggling is covered", tools==[]) → forma
   genérica como R2. 1 amostra.
3. **Regra no validate_sft.py**: "descrição só cita tools declaradas"
   (lista fechada: list_dir, read_file, bash, powershell, write_file).
4. **Sweep fator-descrição** (teste barato SEM retreino): P5 com a frase do
   list_dir removida das defs × original, temp 0.0 e 0.8. Se loop cair,
   causa achada. Usa models/gguf/...v34_ep2_q8_0.gguf + runners com
   --temperature/--seed/--repeat_penalty (já implementados).
5. **Métrica undeclared-call no runner** (benchmark_battery_v2.py
   evaluate_sample: flag se call ∉ tools do case + taxa no resumo).
6. **Montagem v36**: PURIFIED − quarantine(q2 394 + qp 42 + qp5 5, uniões
   verificar; Q1 NÃO sai — foi reescrito via R2) → N≈2043. Gates (validate,
   overlap incl. P5 que já está no gate). Sync payload (dataset + train).
   Versionar dataset (-r tar!), push kernel, create_kernel_session (ver
   scripts scratch kaggle_*). NUNCA rode sem confirmar N e bytes antes/depois.
7. **Pós-v36**: baixar output, melhor época (pedir log dev-hard ao usuário),
   GGUF Q8, 5 evals + gate Pi.

## 3. Decisões pendentes (usuário ou crítico)

- P4 error-first (37): manter como pedagogia single-erro→troca (minha tese)
  ou converter p/ ENOENT; ablation B arbitra. NÃO reescrever em massa agora.
- Dangling-como-sucesso na inferência: mantive tese conservadora; inverter
  é 1 linha se o crítico insistir.
- R2 uuid que o crítico alega restar: reproduzi 0; pedir o uuid.
- 5 uuids do QP-48 dele que não reproduzi (tenho 43): pedir lista.
- Pi: sampling real enviado (logs do servidor) + descrição real do read_file
  no Pi (cita list_dir? se sim, bug de harness).
- Ablation A/B (8h T4): só se v36 não fechar L4.

## 4. Mecânicas que custaram horas (não redescobrir)

- Kaggle auth: CLI quebra por KAGGLE_USERNAME/KEY mortos no env. Fluxo:
  python Temp/opencode/kaggle_refresh.py → kaggle_token_sync.py → comandos
  via wrapper com KAGGLE_API_TOKEN + remoção das vars (ver scripts
  kaggle_ds_version.py / kaggle_kernel_push.py). Token dura ~3h (401 = renovar).
- Download output: manifesto lista arquivos, mas URLs montadas dão 404;
  usar o campo url assinado (kaggle_signed_dl.py). Downloader ignora filtro
  sem version_label;amak. Sessão falhada anexa output parcial.
- `kernels push` não roda; run = create_kernel_session via SDK
  (ver scratch kaggle_run_v34.py). shape "NvidiaTeslaT4".
- Datasets: upload pula subdirs por default — SEMPRE `-r tar`, senão
  engine/catalog somem e o kernel quebra no assert de marker.
-Bash tool é PowerShell 5.1: sem `&&`, sem `head/cut`, `Select -First N`
  MATA o processo anterior (nunca pipear download longo nisso). stdout
  redirecionado bufferiza (log vazio ≠ travado). Processos longos: WMI
  detached com log + `< NUL`, monitorar por arquivo/log.
- Python inline: SEMPRE prefira arquivo scratch; quoting PS×Python quebra
  (`\\` vira `\` etc). Erro JSON "Expecting ','" em staging quase sempre é
  backslash: `\v \p \r` em arguments (use chr(92) ou json.dumps programático).
  `re.sub` interpreta `\n` no replacement (use função lambda).
- Merge: backup .bakN automático; merge aceita N arquivos de uma vez;
  uuid duplicado = aborta (bom). Staging entra SÓ via merge_staging.py;
  reescrita de linha fundida = edita PURIFIED com backup + valida + documenta.
- validate_docs.py só checa docs/**/*.md + READMEs, padrões: file://,
  scripts/*.py, scratch/*.py, data/*.jsonl, kaggle_*/. NÃO cobre .json
  (eval_results pode mover com update manual das refs).
- Overlap: assinaturas = números + literais quoted; cego p/ prompts curtos
  (P5 passou com 2 sigs!). Complementar com substring de frases distintivas.
- Monitor monitor.exe trava swap; serve .bat usa porta 8080 (conflita com
  servidor detached — mate o meu antes do usuário usar o dele).
- GPU local RTX 3060 12GB livre (treinos no Kaggle). llama-cpp-python
  carrega/descarrega por célula; servidor + sweep cabem juntos (~6GB).

## 5. Mapa de arquivos

- Dados vivos: data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl (+
  .bak21–.bak29 cadeia; .bak22 é backup redundante de --help errado).
- Congelados: V1 + ultradata-16k (não mexer); evals congelados (3 baterias).
- Staging: v31/ (7 fundidas, registro), v35/ (t_nav1-4, fundidas), quarantine/
  (q1 superada por R2; q2, qp, qp5 = filtro v36).
- Harness P5: data/eval_nav_p5.jsonl (24, overlap 0 HARD, baselines v33 19/24
  e v34 16/24 em data/eval_results/p5/).
- Resultados: data/eval_results/{sft_v31,sft_v33_ep2,sft_v34_ep2,grpo,multiturn,sft_old,p5}/.
- Fixtures reais: C:\Logs\talt (T-ALT), C:\Logs\p4t1..p4t4 (P4). NÃO apagar
  (proveniência dos outputs).
- Scratch reutilizável em Temp/opencode: kaggle_*.py (auth/dl/push/run),
  sweep_p5.py + sweep_read.py, *_gate.py (auditos), check_*.py.
- Docs vivos: CHANGELOG.md (atualizar a cada mudança), 13_v35_plano_correcao.md
  (rev 2; falta rev 3 com round atual), data/README.md.

## 6. Regras duras do repo (AGENTS.md)

CWD = raiz sempre. Single-writer no PURIFIED. Backup antes de escrita em
massa. Lógica Python em arquivo, nunca one-liner engenhoso. Validação é
gate (validate_sft + validate_docs após cada lote). Proibido sintetizar
observations; outputs de tool DEVEM ser executados de verdade. Sem treino
local de época, sem PII real, sem commit de .bak/caches. Subagentes só em
staging separados (este turno não usou nenhum).
