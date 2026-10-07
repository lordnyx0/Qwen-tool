# Changelog do Handover e do Dataset

Formato: data — mudança — artefato. Ordem: mais recente primeiro.
Fonte de verdade para o estado atual: este arquivo + `V1_MANIFEST.md` (congelado) +
`09_v2_roadmap.md` + `10_v2_plano_acao.md` + `11_v3_plano.md` + `12_v3_1_plano.md`
+ `13_v35_plano_correcao.md` (trabalho em curso).

## MODELO GENERIC — 2026-10-06 (trilha generic primeiro)
* 2026-10-07 — **Mirror Kaggle syncado (correcao segura):** `Kaggle/kaggle_sft_data/curriculum_3000_tool_sft.jsonl` 3427→3211 (igual ao PURIFIED; proveniencia anterior murky, agora espelho verdadeiro). Demais arquivos do bundle (train script, rewards 205 linhas, engine.tar, qwen3loop_python) pinados a kernels lancados — intocados. Push remoto (`datasets version`) só com nova run.
* 2026-10-07 — **Lote auditoria externa (a–e):** untrack `engine.tar`x2 + `Programa/build/` + `.pi/` (gitignore: `*.tar`, `Programa/build|dist`, `.pi/`); `torch.load` com `weights_only=True` em 5 sites (probes); requirements += PySide6/peft/hf_hub/gguf/pandas/polars; `classify_effect` reescrito (edit/move/cp/mv/sed/chmod/curl/wget, redirect sem espacos, `execute_python` puro vira read_only; 12/12 casos do relatorio OK); harness `realpath`, `renamed`→família `rename`/`unlink`, `PY_BLOCK` += open-a/pathlib/eval/exec; `BASH_BLOCK` sem `\bread\b` (falso positivo); CI push `.github/workflows/validate.yml` (validate_sft + validate_docs, contents:read). Nao feito (exige GUI/container ou decide deploy): mirrors Kaggle, itens GUI/threads, `cat /etc/passwd` (allowlist = container).
* 2026-10-07 — **Batch2 aplicado (itens 2+5+6):** PURIFIED 3223→3211 (3 outliers >5000 + 8 greetings + 1 args-mojibake; mantidos echos "Confirmed — N", terse mcp_tr_* e respostas diretas t_notool). Backfill de properties em 473 amostras/662 tools (chaves/tipos observados; decoys nao-chamadas intactas; zero novas mascaras C1b por construcao). `scripts/pipeline/cleanup_batch2.py` (backup `.bak78`), manifestos/views filtrados, stats regen. Item 4 virou auditoria: `scratch/audits/audit_retra_flags.py` (só relatório, 210 true-sem-evidência concentrados em s5 authored + 178 false-com-evidência; líquido 431; SEM flip — flag carrega intenção autoral). Gates zerados, dedup exit 0, docs 0 quebras.
* 2026-10-07 — **Fill system+ids no PURIFIED:** 745 systems (modal por pilar) + 870 ids (`call_001`, largura dominante; tcid por pareamento posicional, 0 orfaos, nada sobrescrito); 960 linhas retokenizadas + `token_len` syncado nos manifestos. `scripts/pipeline/fill_system_ids.py` (dry-run + `--apply`, backup `.bak77`). Validador endurecido: `sem_system` vira gate (staging em voo tem 100%), ids como info nao-falha; `merge_staging.py` exige system. Gates: tudo zerado, dedup exit 0, docs 0 quebras.
* 2026-10-07 — **Dedup aplicado no PURIFIED:** 3981→3223 (668 grupos, 758 removidas; keep retra > catalogo pequeno > uuid; retra 15.9%→14.4%, de volta ao teto). `scripts/pipeline/dedup_purified.py` (dry-run + `--apply`, backup `.bak76`), manifestos global/tracks e views filtrados por uuid, stats regenerados. Drop concentrado em `purified_canonical` (-392); `v37_small` preservado (523). Gates: `validate_sft.py` zerado, `check_duplicates.py` exit 0, `validate_docs.py` 0 quebras. Sync Kaggle pendente.
* 2026-10-07 — **Dedup auditado (script do terciario promovido):** `scratch/audits/check_duplicates.py` (fix encoding manifest + flag opt-in `--prefer-retra`; resto verbatim). Leitura no PURIFIED: 668 grupos / 1426 linhas (35.8%) no modo user+calls = 3223 unicas (meta ~2500 do terciario nao fecha com a propria conta); 799/1838 no user-only. 618 grupos cruzam fontes; removiveis concentrados em `manual_v37_small` (554/554!) + `v39_mt` 66 + `v38_beh` 65 — dedup cego mata o slice de catalogo pequeno do Pi. Nenhum `--write-clean` executado; merge só via staging.
* 2026-10-07 — **MCP canonico (decisao do dono):** catalogo 15→27 (`data/tool_catalog/tools_schema.py`: suite 2b com 12 MCP + aliases antigos mantidos; `DOMAINS` workspace 17). `build_mcp_table.py` com `ours`=MCP + tabela regenerada (12 covered, 1 nominal). `harness_local.py` +12 runners testados em smoke, `agent.py` EXEC, `run_pi_gate.py` list_ok aceita `list_directory`, `validate_sft.py` MENTIONS estendido, `02` suite 2 e `data/README.md` atualizados, espelho Kaggle syncado. Narrow canonico 17.4%→45.2% (generic 93.6%, mixed 56.3%, abstention 0 por construcao). Gates: `validate_sft.py` zerado, `validate_docs.py` 0 quebras. Pendente: sonda 25 e `run_pi_gate` ainda declaram `read_file/list_dir` (holdout de transferencia, veredito pendente).
* 2026-10-07 — **Pack terciário dataset integral p/ chat:** `handoff_terciario/` com 2 arquivos (dataset_manifest.jsonl 3981 linhas + PROMPT_E_REGRAS.md), sem resumo agregado, sem acesso externo. `validate_docs.py` 0 quebras.
* 2026-10-07 — **Pack terciário compactado p/ chat:** `handoff_terciario/` reduzido a `PACK_UNICO_CHAT.md` único (6.5 KB, autocontido, sem acesso externo; resumo global 3981/generic 475/narrow 2165 + regras 02/07/08/09/V1/14 + prompt fechado sem-bash). `validate_docs.py` 0 quebras.
* 2026-10-07 — **Pack terciário generic vs narrow:** pasta `handoff_terciario/` (cópias read-only de `data/split/` + docs 02/07/08/09/V1/14 + exemplar + `PROMPT_TERCIARIO.md` fechado sem-bash). Fonte de verdade segue no repo; `validate_docs.py` 0 quebras.
* 2026-10-06 — **Supervisão profunda ligada PERMANENTE (engine+config+train):**
  `sft_final` (base, = HF `Lordnyx/qwen3loop-0.6b-sft-deep-supervision-v1`) já trazia
  `loop_loss_weights=[0.4,1.0]` no config, mas o branch de loss tinha guarda
  `not enable_double_loop_split` → com split=True (base e v38) a CE intermediária NUNCA
  disparou em nenhum treino. Agora: exits = 3 snapshots de segmento (prof. 21/35/49) + saída
  final peso 1.0; snapshots passam pela RMSNorm final só na projeção auxiliar (sem ela,
  CE≈180/logits 1176, risco NaN em bf16; com ela, loss ponderada 9.19 vs 7.81 single).
  Smoke em GPU local com pesos reais: exits=3, gradiente flui, branch engaja.
  Trava tripla: default config `[0.4,1.0]` + default train `0.4,1.0` + asserts fail-fast
  (pesos vazios) e no step 0 (exits>1). Engine sincronizada (`engine/` =
  `kaggle_sft_data` = `kaggle_grpo_wsps`, hash único).
* 2026-10-07 — **GRPO generic: teste FALHOU (reward 2.19→1.43, nocall→0.58, sem pesos salvos).**
  Diagnóstico: KL ancora só a saída final; mids com policy-grad desligado (GAMMA 0) e distill
  p/ final móvel = deriva livre (ppl_loop1 em spikes até 323). Fix: `MID_KL=0.10`
  (`mid_policy||mid_ref` congelada, mesmo `mid_idx`, `--mid_kl` + asserts fail-fast;
  matemática checada). Expansão p/ generic single VETADA pelos números.
* 2026-10-07 — **S8d mergeado (72 singles grounded):** coleta Actions 137 calls → 72 goldens
  (58 ps + 25 bash − 11 dupes); `s8d_authored.jsonl` (users 72/72, gates zerados).
  PURIFIED 3909→**3981**, manifest +72 (`.bak20`). Generic: 546; linux: 120 (58/62).
* 2026-10-07 — **qp17: python fora do generic (194 uuids, PURIFIED 4103→3909).**
  Narrow tinha zero python (intacto); mixed mantém 8. Gates zerados.
* 2026-10-07 — **S8d disparado (coleta de singles shell):** 137 calls (92 ps + 45 bash:
  bare originais + 8 stories novas em fixtures) p/ goldens de singles (SFT + RL).
* 2026-10-07 — **GRPO v7 Pi-5: 0.0/5 (pior que SFT 2.0/5) — linha GRPO generic ENCERRADA:**
  ckpt-50 mergeado (`generic_grpo50_q8_0`); policy virou regurgitadora de goldens S8b/S8c
  (`/tmp/s8cap`, `/tmp/s8b05`... em toda resposta, zero aderência ao prompt). RL amplificou
  os priors memorizados do SFT em vez de corrigi-los. Trajetória já dizia (reward↓, nocall↑).
* 2026-10-07 — **GRPO v4 (currículo só-alcançável + mid-KL):** `grpo_generic_v2.jsonl`
  (163: wsps 75 + abstain 30 + discrim_shell 30 + shell_multi 28), sampler adaptativo.
* 2026-10-07 — **GRPO v5/v6 ERROR (mount do dataset) + v7 RUNNING em dataset novo:**
  `ERRORED_MOUNTING_DATASET` grudado na versão (2 retrys falharam <60s). Dataset fresco
  `grpo-generic-data2` (flat + engine.tar; ingestion extraiu `engine/` sozinha) + kernel
  apontado p/ ele. Notebook com tar-loop e busca em 3 prefixos (à prova da quirk).
* 2026-10-06 — **GRPO do generic DISPARADO:** policy `generic_sft` epoch-2, currículo 149
  (wsps 75 + pi abstain/discrim/multiturn shell), goldens 522, juiz plumb, 100 steps.
  Datasets novos: `grpo-generic-data` + `generic-sft-model` (GGUF juiz fora, plumb decide).
  v1 ERROR (engine ausente no `create`); v2 RUNNING com busca em 3 prefixos.
* 2026-10-06 — **Gate Pi-5 do generic: 2.0/5 (missing+port) vs v38 2.0/5 (mkdir+list).**
  No caminho: ponte bash do harness quebrada (Git Bash com PATH vazio → `mkdir/ls: command
  not found`; todo gate anterior media o harness, não o modelo — corrigida com PATH injetado).
  Gaps reais do generic: regurgita comandos S8b (`mkdir /tmp/s8cap` em vez de `asfkap`),
  shelja chitchat (o prompt exato estava no abs60!), repete `ls` sem fechar. GGUF:
  `unrolled_modelo_qwen3loop_tool_sft_generic_sft_q8_0.gguf` (F16s intermediários limpos).
  primeira bateria do modelo generic — 5 tarefas (tarefa python removida, fora do escopo),
  fixtures por task (`serve_agent.bat` agora existe no root: tarefa 3 virou passável),
  scoring por regras, log em `runs/agent_gate_pi5_*.jsonl`.
  Baseline v38_ep2 Q8: **1.0/5** (só list). Alvo do generic: ≥3.5/5.
* 2026-10-06 — **TREINO GENERIC v1 FALHOU + v2 DISPARADO:** v1 ERROR em 20s —
  `kaggle datasets version` não subiu subdirs (`qwen3loop_python/` ausente → assert do
  marker). Fix: re-push trouxe tudo (`engine/` solto funciona; `engine.tar` virou redundância
  local). Kernel v2 RUNNING (mesmo plano: 534 amostras, epochs 3, base `sft_final`).
* 2026-10-06 — **v2 ERROR (script errado!) + v3 DISPARADO:** o `os.walk` do notebook achou
  `train_sft_tool_qwen3loop.py` no bundle do modelo PRIMEIRO (variante velha de 420 linhas,
  sem `--log_steps/--val_frac/--dev_eval` → argparse exit 2, treino nunca começou).
  Fix permanente no notebook: script = `dirname(data_path)` (mesmo pacote, sem loteria).
  Kernel v3 RUNNING.
* 2026-10-06 — **qp14:** removidas as 15 `technical_api` do generic (6 `t_err` + 6 `t_un` drills
  + 3 `t_sml` brinquedos; nenhuma API de domínio era chamada de verdade). PURIFIED 3977→3962,
  `apply_quarantine.py` (novo, canônico: remove de PURIFIED+manifest, retokeniza, stats).
  `validate_sft.py` + `validate_docs.py` passam.
* 2026-10-06 — **Enxerto abstention:** `data/split/views/recipe_generic_abs60.json` — 60 UUIDs do
  track abstention p/ view de treino (30 `t_abs` sem tools nos pilares ws/ps/bash + 20 `v38_beh`
  + 10 `v37` factuais com catálogo). Sem alterar PURIFIED.
* 2026-10-06 — **Scaffold S8:** `data/staging/s8_generic_multi_scaffold.json` — 70 slots multi
  generic (catálogos 8–15) a partir de `shell_golden_table.json` verbatim (52 R-R, 5 R-M, 8 M-M,
  4 M+R-ind, 1 R-R-R). Turnos user/assistant = redação humana pendente.
* 2026-10-06 — **S8c: `terminal_linux` a 100 (38 single / 62 multi).** Coleta Actions
  `s8c-golden` (125 calls bash, 123 ok, zero vazios) + fix-run da story 46 (`diff -r` que difere
  dá exit 1 por desenho — fix captura `diff-exit-equals-$?`, golden exemplar).
  `s8c_golden_table.json` (120 keys RAW), `s8c_authored.jsonl` 73 amostras (users 73/73 únicos,
  gates zerados). PURIFIED 4030→**4103**, manifest +73 (`.bak18`).
  **Generic: 668 (406 single / 114 retry / 148 multi)**; treino com enxerto abs60 ≈ 728.
* 2026-10-06 — **S8 authored + merged (regra anti-síntese derrubada pelo dono):**
  `data/staging/s8_authored.jsonl` — 70 amostras (users 70/70 únicos, thinks reusados ≤5×,
  closers ecoam observation verbatim; catálogos 8/12/15/10 com decoys reais do pool generic).
  No caminho: bug de catálogo sempre-8 (`mcp_tools.json` é lista, pool de decoys vinha vazio —
  corrigido p/ pool de 1156 defs) + 1 `desc_decl` (doc cross-ref, decoy trocado).
  PURIFIED 3960→**4030**, manifest +70 (`sync_manifest_s8`, `.bak17`), `validate_sft` zerado.
  **Generic final: 595 (single 381 / retry 114 / multi 100)** + 60 abs da receita = treino ~655.
* 2026-10-06 — **qp16:** `t_sml_0390/0391` dupes exatos de `t_err_0026/0030` (PURIFIED 3962→3960).
  **Backfill `is_retrajectory`:** 283 Nones → 13 TRUE (recovery genuíno `t_beh_*`, revisado 1×1)
  + 270 False; manifest dizia False p/ os 13 (default de build) — sincronizado.
* 2026-10-06 — **Achados generic (pente-fino):** `t_run_*` inocentado (round-trip real confirmado);
  pilar python 100% math-drills (0 coding real); `terminal_linux` 13 singles; 28 calls em nomes
  legados no `workspace_code` (veredito pendente); v38_beh com outputs authored (não golden).

## WORKBENCH MODULARIZADO & DESHARDCODADO — 2026-10-05 (Multi-SO, Portável, Otimizado)
* 2026-10-05 — **Deshardcoding completo:** Removidas todas as referências rígidas a caminhos
  pessoais (`Nyx`, `Temp/opencode`). Resolução de temporários portável (`paths.get_log_dir()` em `.cache/logs/`),
  busca dinâmica de snapshots no cache HuggingFace (`find_hf_model_snapshot`) e âncoras universais de raiz.
* 2026-10-05 — **Multiplataforma de Processos:** Substituído script Windows-only WMI por
  `launch_detached_process` (`subprocess.Popen` nativo com `CREATE_NEW_PROCESS_GROUP` / `DETACHED_PROCESS`
  no Windows e `setsid` no POSIX). Monitoramento de PIDs e `kill_process_tree` multiplataforma.
* 2026-10-05 — **Configuração Declarativa & Auto-discovery (`config.py`):** Suporte a `workbench.json`
  para desacoplar nome do projeto, caminhos de datasets, esquemas, jobs e evals. Auto-sintetização de
  manifesto para projetos externos (ex: `svg_generator`).
* 2026-10-05 — **Otimização de Performance:** `SplitsTab` refatorado para `QTableView` virtualizada
  com `SamplesTableModel(QAbstractTableModel)`. Carga e filtragem de 3.447 amostras em < 0.1ms (vs ~500ms
  no `QTableWidget` com 24k itens instanciados). Spec do PyInstaller enxuta com exclusão de módulos Qt órfãos.
* 2026-10-05 — **Features Especializadas:** Renderizador vetorial SVG (`QSvgWidget`) dinâmico no visualizador
  de splits para datasets gráficos (`svg_generator`); seletor de projeto `Arquivo -> Abrir Pasta do Projeto... (Ctrl+O)`
  e varredura recursiva de markdown na aba Docs.
* 2026-10-05 — **Filtro de Efeito Colateral:** Adicionado 2º filtro na aba Splits (`read_only`, `mutating`, `none`)
  e flag `--effect` no `scripts/pipeline/make_view.py`.

## DPO ENTERRADO — post-mortem 2026-10-05 (caminho fechado, checkpoints apagados)
* v38 103 → v3 86 → v4 88 → R4ep1 85. Quatro rounds, o método perde ~15
  pontos em qualquer dado/ancora/beta. SHIP segue v38_ep2.
* Erros cometidos, para não repetir:
  1. `seq_logps` off-by-one (2 rounds treinando em auto-predição).
  2. Detector com furo Unix ("No such file", REFUSED fora) → chosen com 38%
     de calls falhadas.
  3. "Limpo" confundido com "cumpre a tarefa" (detectores medem forma;
     fulfillment exigiu revisão manual 1×1; 29/62 divergentes inválidos).
  4. Confound de comprimento (chosen 0,7 calls vs rejected 3,9) — modelo
     aprendeu brevidade = undertrigger (L2/L3 caíram, L1 intacto).
  5. Âncora SFT presa ao curado = mesma direção da distorção (soma-zero:
     up no curado é down nas alternativas válidas).
  6. Beta 0,2 + gradientes corrigidos = overshoot; sem saves por época até
     o R4 (comparação ep1×ep2 impossível nos rounds 1–3).
  7. Saldo honesto final: 17 pares length-matched + tarefa-cumprida em
     4730 rollouts — pouco para DPO. Preferência sem grounding não instala
     juízo. Próximo: SFT (dado), não DPO.
* Restos: `models/_quarantine/` (dpo_v3, registro) mantido; demais
  checkpoints/Q8 DPO apagados (~9GB).

## DPO round4-bestofn (v38 + 55 pares on-policy verificados) — RUNNING 2026-10-05
* 2026-10-05 — Kernel v5 lordnyx0/qwen3loop-tool-dpo (45 treino + 10 held,
  `dpo_*_r4c.*`): base v38, beta 0,1 + `sft_anchor 0,5` + `loop_aux 0,2`,
  shift fix, **saves epoch-1/epoch-2** (comparação held-out por época =
  trava de overfit). Stop rules: bateria ≥100, gate sem regressão, held-out
  subindo. Dataset em `data/staging/dpo_round4/` (+ r4b intermediário).
* 2026-10-05 — Pares R4c: mineração best-of-n 300×10 + top-up 173×10 (4730
  rollouts via 8084) → detectores (furo Unix tapado: "No such file",
  "cannot access", REFUSED, syntax error) → chosen estrito (zero vícios +
  100% calls `ok`/abstenção correta) → **revisão manual 1×1 dos 62 sem
  overlap: 33 válidos** (17 abstenções + 16 alternativas-válidas) + 26
  overlap = 55 pares. Achado: 38% das calls "limpas" tinham falhado
  (buraco do detector); fabricação e desobediência filtradas.
* 2026-10-05 — dpo_v4 avaliado (round3: beta 0,1 + âncora): bateria **88/121**
  (+2 vs v3, −15 vs v38), gate 2+1parcial (math recuperado: execute_python →
  62748517). Âncora mitigou, não curou (ancorava o curado = mesma direção
  do DPO). Decisão: best-of-n on-policy (massa para acertos próprios).
* 2026-10-04/05 — Diagnóstico loop (lens por snapshot, 24 prompts, base vs
  v38): snap1 agree 70,8% idêntico; loop intacto (SFT não degradou).
  Refinamento é polidor (rank 478→393, flips 1/101), não revisor.
  seq_logps do DPO estava off-by-one (labels[t]~logits[t]); corrigido +
  `--loop_aux` e `--sft_anchor` (DPO+) no `train_dpo_tool.py` (smoke OK).
* 2026-10-04 — qp8: quarentena `t_nav2_0039` + `t_navb_0039` (PLACEHOLDER_E39,
  duplicata alias); PURIFIED 3633→3631, `.bak51`, validação zerada. Hunt
  em `data/` (50 arqs): resto legítimo (E302, BMW E46, EDINET, TODOs reais).

## DPO round2 (v38 + R2 + beta 0.2 + loop_aux) — ENTERRADO 2026-10-04
* 2026-10-04 — dpo_v3 (Kaggle lordnyx0/qwen3loop-tool-dpo, 256 pares R2, shift
  fix): margens treino 0,79 / held-out 1,18 (generaliza) mas bateria 86/121
  (-17 vs v38: L2 36→31, L3 19→16, L4 12→9) e gate-6 1+2parciais (math quebrou:
  `echo $13**7` → "3 million"). Causa: DPO puro sem ancora de capacidade +
  gradientes corrigidos (shift) + beta 0,2 = overshoot. Stop rule (piso 100)
  violada. Artefatos em `models/_quarantine/` (fora da linhagem).
  Proximo: v38 + beta 0,1 + ancora SFT (DPO+).

## V3.1 (treino v33 — epoch-2 eleita, NO-SHIP 3/5)
* 2026-10-01 — Organização (pedido usuário): `Kaggle/_legacy/` recebe
  kaggle_grpo_v3 (1,1GB) + kaggle_kernel_grpo (sem referências; model_src e
  kaggle_bundle intocados por dependência viva); árvore do README reflete
  `Kaggle/` + pack_agentic. Docs 104/0.
* 2026-10-01 — Casa organizada: tudo Kaggle em `Kaggle/` (payload, kernel,
  outputs; pasta já existia e recebeu o conteúdo — integridade verificada:
  notebook v37, payload N=3427). Incidente honesto: criei `Kagle/` e o destino
  real foi `Kaggle/` (nome correto, pré-existente). Refs atualizadas (46),
  docs 104/0. Exceção registrada: 1 path em V1_MANIFEST atualizado por
  mudança de local (dado V1 intacto). .bak antigos purgados (26 arqs, 252MB;
  cadeia 39-50 mantida).
* 2026-10-01 — DPO ep2: margens dobraram (+0,144, overtrigger +0,22) mas gate
  parado (~1/~2; porta 0.0 confabula 8000). Link margem→comportamento quebrado
  nesta escala. DPO ENCERRADO (2 eps, bateria 102 intacta).
* 2026-10-01 — DPO v1 (242 pares vice-real, beta 0.1, 1ep, local): bateria 102
  (abstinência 15/20 mantida); gate t0.0 ~1,5 (dup 44%!) / t0.8 ~1,5. Não
  moveu políticas; 0.0 até piorou em loops. Hipóteses: beta/épocas, pares
  fáceis demais, ou DPO fraco p/ 0,6B. NÃO SHIPPAR; tuning ou parar.
* 2026-10-01 — t_mt 66 fundidas (N=3.633; overlap real 179 HARD genéricos,
  inalterado); mix3 pacote: bateria 98, sonda 13/25, multiturn 11/12, P5
  18/24. Mais recusa não ajudou o gate (mix2 3,5 segue melhor).
* 2026-10-01 — CORREÇÃO de processo: meu `check_eval_overlap.py FILE
  --baseline` NUNCA checou o FILE (--baseline ignorava o arg; O=0 era vácuo).
  Script corrigido (file sempre checado quando dado). Check REAL no PURIFIED:
  179 HARD, quase todos números genéricos ('100', '500') em user-msgs —
  ruído de assinatura, não contaminação verdadeira; gates de staging (os que
  valem) sempre foram reais. t_mt 70→66 (4 HARDs removidos). Lição: desconfiar
  de gate verde sem linha de arquivo no output.
* 2026-10-01 — mix3 pronto (585: 220 pos + 60 pares + 165 recusas (30 t_cal-b,
  90 t_irr, 5 manuais, 40 small 2-4 tools) + 140 cirúrgico; 28,2% recusas).
  Checkpoints falhos apagados (11,2GB). Validate verde. Aguarda GPU livre.
* 2026-10-01 — mix2 (v38+545 c/ pares, LR 5e-6, 1ep, dev 83,3): bateria 98,
  sonda 13/25 (python 5/5), multiturn 12/12, P5 18/24, gate-harness 3,5/6
  (porta+pycalc PASS c/ evidência). 4/5 barras (L4 12/20). Melhor no gate,
  perde bateria/P5/abstinência p/ v38. Manifest em data/staging/v38_mix2/.
* 2026-10-01 — Mix local (v38_ep2 + 400 replay/140 cirúrgico, LR 8e-6, ep1
  86,7% dev): bateria 94 (abstinência 9/20, REGRESSÃO), gate ~2/~2 (hi
  corrigido, porta 8080→80, pycalc intacto). LR move comportamento mas sem
  controle de direção; replay precisa quota de recusas. NÃO SHIPPAR.
* 2026-10-01 — Harness próprio criado (scripts/agent/agent.py: 5 tools do
  treino, git-bash real, breaker de retry idêntico, temp 0.8/rp 1.1, log
  JSONL). Aceite v38: 1 PASS (asfkap) + 3 parciais; breaker disparou 3×;
  pycalc nunca alcança execute_python, hi over-triggera, paths inventados.
  O pack v38_beh (107, staged) mira exatamente isso.
* 2026-10-01 — Dataset v39 no HF (privado): Lordnyx/qwen3loop-tool-sft-v39
  (N=3.567 + card com projeto, categorias, gates e limitações).
* 2026-10-01 — Cirúrgico local (v38_ep2 + 140, LR 1.5e-6, 2 épocas, 20min):
  dev ep1 81,7 > ep2 80,0; bateria ep1 101 < v38 103, ep2 100; gate ep1/ep2
  sem correção (pycalc ecoa, porta confabula 5357/80, hi@0.8 over-trigger).
  Veredito: patch NÃO moveu as políticas (35 steps fracos demais e/ou sulco
  fundo demais). NÃO SHIPPAR cirúrgico; devolver matriz ao crítico.
* 2026-10-01 — Pack cirúrgico v38_beh em STAGING (107, sem merge): 75 upsamples
  (r2/r3/r5/r6), 23 run-code c/ stdout real (vazio/expr-sem-print fora: sem
  output p/ supervisionar), 10 no-tool seguros; regra generica refinada
  (vazio/conhecido-vacuo; base segue verde); 1 HARD removido. Gates verdes.
* 2026-10-01 — HF publicado: Lordnyx/qwen3loop-0.6b-tool-calling-sft-v38
  (público; safetensors ep2 + Q8 + card com métricas, EN-only declarado,
  limitações conhecidas).
* 2026-10-01 — Gate próprio v38 (loop fechado modelo+harness real, sandbox):
  t0.0 ~1,5-2/6, t0.8 2,5/6. Separação limpa: hi over-trigger@0.0 vs recusa@0.8
  (sampling, consistente cross-harness); bugs model-side nos dois temps e nos
  dois harnesses (pycalc ecoa literal, paths inventados, retry idêntico);
  harness molda trajetórias (REFUSED, toolset). Pi do usuário: 2,5/6 (t0.8).
* 2026-10-01 — v38_ep2 avaliado (Q8 local, temp 0.0): bateria 103/121 (85,1%),
  L1 30/30, sonda 13/25 (python 5/5, abstinência 0/5), multiturn 11/12,
  P5 22/24 (t0.0 e t0.8, undecl 0). 5 barras: 5/5 PASS (L4-abstinência 15/20).
  BFCL: simples 81,1 / múltipla 88,7 / irrelevância 60,0. Primeiro SHIP
  candidato; ressalvas: probe-abstinência 0/5, multiturn -1, Pi gate pendente.
* 2026-10-01 — v38 montado p/ treino (épocas=2, sem 3ª): import-test do train
  ok (C1/C1b/C1c+C3 presentes); saves com rotação keep=2 + epoch-{1,2}
  preservados (sem risco ENOSPC além do já corrigido); notebook epochs 2 +
  mask flag; payload sync (N=3.427, mask 67); dataset v38 versionado
  (17.243.162 bytes); kernel v37 pushado. Run contigo (2 épocas, patience 1).
* 2026-10-01 — Resposta auditoria externa: ids renormalizados (267→0) +
  tool-result names alinhados via tcid (201→0; bug meu nos aliases + 2 t_nav4;
  .bak49); C2-props preencheu 121 defs vazias c/ evidência; C1c mascara
  call-seguida-de-erro exceto P4/aliases (Dataset B sem dividir; teste e2e
  PASS); re-medição: args-fora-schema 187 genuínos (3,3%, cobertos por
  C1b/C1c no treino) + 1.197 estruturais-vazios. N=3.427, gates verdes.
* 2026-10-01 — Resposta ao relatório: 52 executores strippados dos t_irr
  (contagem bateu exata; +10 descs p/ genérica; .bak46/.bak47); 640 cópias
  small-catalog t_sml_* (1ª+2ª fatias, 29+10 HARD removidos); think t_irr em 3
  variantes (56/50/50; .bak48); trainer tolera pillar ausente (default rec=1).
  N=3.427, 1-4 tools 25,9%, gates verdes.
* 2026-10-01 — Pré-auditoria: finais t_irr auditados (0 issues); token_length
  recalculado (>5.000: 2, purified_1715/1821, fora do treino por substituição
  silenciosa); mask 68→67 (purified_0257 saiu via qp7; nome do arquivo mantido);
  payload re-syncado (N=2.787). Gates verdes.
* 2026-10-01 — Finais t_irr reescritos (10 padrões, propósitos reais do
  catálogo, distribuição 15-20; padrão-8 nunca disparou = overlap-zero total,
  evidência da triagem); revisão encontrou 2 remoções (t_irr_0017 base64-exec,
  t_irr_0097 recon×bash = falsa-recusa; screen D2 limpo no resto; .bak45).
  N=2.787, gates verdes.
* 2026-10-01 — Lote irrelevância autorado sob delegação (juízo delegado pelo
  usuário, auditoria do crítico ex post): 300 pares minerados → telas B-desc/
  C-syn/D-sens + 30 catalogos-ausentes → 165 → 7 HARD de overlap removidos →
  158 t_irr_* fundidos (requests reais, catálogos reais 8-20 tools, finais em
  template sem eco). N=2.789, validate (13) + overlap + schema-gate verdes.
* 2026-10-01 — v37 montado (sem run): payload sync (N=2.631, train c/ C1/C1b/
  C3+regex, mask JSON), dataset versionado (14.942.459 bytes); kernel push
  BLOQUEADO (quota GPU semanal 30h esgotada — push+run juntos qdo voltar).
* 2026-10-01 — Rodada do crítico 2: token_length recalculado (tokenizer real;
  stale 1.855/2.631, med 1.084, >2.048: 323, >4.096: 9, >5.000: 1; .bak40);
  distratores v2 (strip 19.424, alvos ≤2, head Jaccard>0-top3, resto rotação
  uniforme: top50 17,5%, catálogos-c/irmã 14,5%, mesmo-vocab 22%; .bak41);
  300 pares irrelevância minerados (223 catálogos); 28 t_cal*b p/ forma
  genérica (.bak42; 6 resíduos já cobertos pela C3); aliases v2 (t_nava fora,
  t_navb 200 c/ vocab 30/30, reais em 15; .bak43). N=2.631, gates verdes.
* 2026-10-01 — qp7 removidos (25 purified c/ ruído de tipos, decisão usuário;
  backup .bak39; N=2631; schema-gate 3.668 calls zerado). Validate (13) +
  overlap verdes.
* 2026-10-01 — ROOT CAUSE + replay distratores: o merge_staging.py tinha prune
  de "declaracoes mortas" no corpo principal que apagou os 1.596 catálogos
  expandidos no merge dos aliases (falha minha de verificação: conferi N mas
  não tamanhos). Prune removido do merge (tools-não-chamadas são sinal legítimo:
  distratores, irrelevância); expansão reaplicada deterministicamente
  (1-tool 0%, ≥3 73,5%, max 20; backup .bak38). Nota honesta: defs
  sem-properties 48,9% por efeito-denominador (cópias distratoras; defs das
  tools chamadas intactas). Validate (13) + overlap verdes.
* 2026-10-01 — Harness local real p/ autoria 3-4: fixture_fs (15 arqs, mtime
  fixo, binario/dir-vazio/erros), harness_local.py (read_file/list_dir/
  powershell/execute_python; jail+blocklist+timeout; list_dir ordenado),
  fixtures_exec.jsonl (29 pares verbatim: 22 ok + 7 erros/BLOCKs);
  validate_calls_schema.py (gate p/ lotes novos; legado: 34 violacoes tipo em
  25 purified -> quarentena/qp7 reversivel, sem remocao); 4 def/call trocados
  meus corrigidos (t_nav_0013 args, t_nav2_0047 def + copias alias; .bak37).
  N=2656, gates verdes.
* 2026-10-01 — Correções finais do crítico (tudo local): C2-required inferido
  por uso (unanimidade, n≥3, key-set único; 26 nomes, 173 defs; backup .bak36;
  divergências required são pré-existentes multi-key, puladas) + regra reqsubset
  permanente (required ⊆ properties); C3 no treino p/ os 68 residuais
  (--mask_first_turn_file, default off; data/mask_first_turn_68.json; guard
  __main__; notebook cell4 c/ flag; JSON copiado ao payload; teste unitário
  e2e PASS c/ tokenizer real). Validate + overlap + docs verdes. Falta (quota):
  sync payload, versionar, push kernel, run.
* 2026-10-01 — Falsa-irrelevância 257: definição reproduz 256 (251 purified+5
  manual; delta 1 = quarentena qp5/qp6 posterior à medição). Fonte
  Tool_Use_049587 confirma hipótese: negação-then-update com tools finais
  renderizadas flat (artefato herdado, não da purificação). Colapso p/ turno
  único nos 183 com meio só-chatter (backup .bak35); 68 com info real no meio
  ficam como resíduo (mascarar no treino ou revisar). 5 manual_v31 mantidos
  (esclarecimentos reais). Bug t_nav2_0019 achado: tnav2_fix.py:32 (append cego
  de BSH); regra dupname permanente no validate (dispara 24/24 no .bak33).
  required reconciliado (1.213 sem required não-vazio = 731 vazias + 482).
  EN-only decidido → declarar no model card. Validate + overlap verdes.
* 2026-10-01 — Correções especialista (sem Kaggle, quota baixa): qp6 23
  overloads nome-duplicado removidos (archival_memory_search x22 + send_email;
  stageados reversível; backup .bak33) + dedup bash em t_nav2_0019 (meu slip T2);
  DISTRATORES nos 1.596 single-tool (catálogos 5-20, metade similar/seed-uuid,
  pool 1.225 limpo; backup .bak34); ALIASES P4 200 cópias t_nava_* (cat_file/
  shell/pshell/list_files/save_file, staging v37_alias, merge ok); BFCL v34_ep2
  em data/eval_results/bfcl_v34_ep2.json (simples 75,7 / múltipla 80,3 /
  irrelevância 48,0 / multi-turn 100 / paralela sem casos); scaffolds itens 2-4
  em data/staging/v37_templates (+README). N=2656, validate + overlap verdes.
* 2026-10-01 — v36 PRONTO p/ run (usuário lança): PURIFIED N=2479 (−qp5 5;
  q2/qp já fora desde v35; 4/5 qp5 c/ repeat consecutivo idêntico). Dataset
  versionado (curriculum 6.277.351 bytes, train 34.056 c/ regex apertada),
  kernel v36 pushado (notebook inalterado, datasets sem pin). Backup .bak32,
  validate + overlap (167 sigs, incl. P5) verdes.
* 2026-10-01 — Métrica undeclared-call no benchmark_battery_v2.py (flag por
  case + taxa global/por categoria no resumo e no JSON); sweep fator-descrição
  no P5 c/ v34_ep2_q8 (A=original × B=sem cláusula, t 0.0/0.8, seed 42, rp 1.0):
  A 15/24→19/24, B 14/24→19/24, undecl 2→1 em ambas; ZERO calls list_dir em
  qualquer célula — cláusula da def EXONERADA como causa de loop; achado real:
  alucinação create_dir/mkdir (001/017) + read_file onde bash esperado.
  Reescrita das 266 mantida como higiene (remove affordance falsa).
* 2026-10-01 — R2-15b + gêmea 14b: finais "only toggling/setpoints are covered"
  com tools==[] → forma genérica ("I have no tool for that", estilo t_cal-b).
  Backup .bak31, validate verde. 01b–07b mantidas (padrão "X only Y" deliberado).
* 2026-10-01 — Regra desc_decl no validate_sft.py (descrição só cita tools
  declaradas; lista fechada list_dir/read_file/bash/powershell/write_file):
  disparou 266; reescrita mecânica das 266 (238 read_file→shell declarado —
  powershell Get-ChildItem / bash ls / genérica — + 28 list_dir→cláusula
  genérica; 51+51 ocorrências legais intactas, ex. s5_ws_0001). Backup .bak30,
  validate verde (desc_decl 0).
* 2026-09-30 — Réplica 2 verificada: 864/162/138 (regex antiga inflava erro com
  'error' solto — apertada no treino e no C2); C2 refeito +8 (292; resto é
  dangling-como-sucesso, tese conservadora mantida); 47 required = pré-existente
  + manual-by-design (C2 nunca escreve required, provado); 2 required inválidos
  corrigidos (1 meu t_nav_0047 + 1 legado t_sem_008); R2-restante = 0
  reproduzível (uuid pedido); 5 purified em `data/staging/quarantine/qp5_purified.jsonl`.
  Backup .bak29, validate verde. v35 segue RUNNING (tudo aqui é v36).
* 2026-09-30 — v35 montado e rodando: PURIFIED 2720 -quarentena(436: Q2 394 +
  QP 43) +P4 (200) = **N=2484** (backup .bak28; validate verde; tecnical 50.8%,
  workspace 23.6%, terminal 34); payload sync (dataset + train c/ mask+C1b);
  dataset versionado, kernel v35 pushado, sessao 354431631 RUNNING (~4h).
  Contém: R1/R2, C1/C1b, C2, P4-T1..T4.
* 2026-09-30 — P4-T4 em staging (`data/staging/_archive/v35/t_nav4.jsonl`, 50: flow lib/
  site/jobs/i18n 20 + summaries 15 + edges hidden/empty/missing/deep 15;
  fixtures C:\Logs\p4t4). Gates verdes. **Dose completa: 200/200** (T1+T2+T3+T4).
* 2026-09-30 — P4-T3 em staging (`data/staging/_archive/v35/t_nav3.jsonl`, 50: flow wiki/
  cfg/srv/logs3/csv 20 + summaries 15 + edges hidden/empty/missing/trunc 15;
  fixtures C:\Logs\p4t3). Gates verdes. Dose: 150/200.
* 2026-09-30 — P4-T2 em staging (`data/staging/_archive/v35/t_nav2.jsonl`, 50: flow api
  20 + summaries 15 + edges hidden/binary/deep/empty/missing 15; fixtures
  C:\Logs\p4t2). Gates verdes (validate zero, overlap 0 HARD). Dose: 100/200.
* 2026-09-30 — P4-T1 em staging (`data/staging/_archive/v35/t_nav1.jsonl`, 50 amostras
  manuais c/ outputs reais, fixtures C:\Logs\p4t1): flow ls->read->resumo (20),
  summaries (15), edges truncation/empty/missing (15); simetria bash/ps.
  Gates verdes (validate zero, overlap 0 HARD). Traducao: sem placeholders,
  sem backslash solto (licoes do ciclo).
* 2026-09-30 — R1+R2 aplicados no PURIFIED (backup .bak27): R1 reescreve call_2
  das 14 T-ALT (tentativa variada, erros E3/E4/E5/E12/E13/E14 verbatim) —
  repeats manuais zerados; R2 reescreve finais das 57 Q1 em 8 formas genéricas
  (contradições zeradas). validate verde. QP-7: quarentena nos 7 (nenhum
  legítimo). P4-T1: fixtures + todas as execuções capturadas (roster 50 pronto).
* 2026-09-30 — Eval v34_ep2 Q8: bateria **95/121 (78.5%)** [L1 30/30, L2 35/41,
  L3 18/30, L4 12/20], sonda 11/25 (python 5/5), multiturn **12/12**,
  dev-hard 20/60 (L4 0/15), P5 16/24. Barras **4/5** (só L4 12/20 <14/20).
  NO-SHIP por 1 barra. Trade visível: +40 abstencao puxou L4 9->12 e multiturn
  9->12, mas L3 24->18 e mkdir P5 3/4->1/4 (inclui invencao nova `create_dir`).
  Resultados em `data/eval_results/sft_v34_ep2/` + `data/eval_results/p5/v34_ep2.json`.
* 2026-09-30 — Sweep P5 (22 células) lido: falha 20.8% (0.0) / 34% (0.3) /
  29-31% (0.8); rp quase não mexe. Regra pré-registrada NÃO dispara nenhum ramo
  (0.0 ≤25% ok, mas 0.8 ≥60% falha; dado exigiria ≥50% em 0.0) = seleção
  first-response é ok em 0.0 e pouco sensível a temp — os loops do gate são
  dinâmica multi-turno, não sampling. Núcleo duro em 0.0: 005, 006, 017, 019,
  021 (seleção list/mkdir). Veredito: misto → v35 completo.
* 2026-09-30 — Gate Pi repetido em temp 0.0 (`--temp 0.0`, Pi real via `pi -p`,
  servidor com log): **3/6** (vs 1.5/6 a 0.8): mkdir PASS (pasta criada, sem
  rm), porta PASS (8080), 13**7 PASS; ENOENT/hi!/list FAIL com geracoes
  monstruosas (~12k tokens/turno, determinísticas). Leitura pela regra:
  falha@0.8=75% -> falha@0.0=50%: melhora parcial, mas 50% em 0.0 >= piso 50%
  = MISTO (não decoding-puro) -> v35 completo. Servidor default 0.8 confirmou
  via `--help` (era o temp do gate original); Pi usa o default (loops
  determinísticos = assinatura greedy).
* 2026-09-30 — C2 aplicado: `scripts/pipeline/infer_tool_schemas.py` (tipos por
  valor, só de sucesso, p1 amostra + p2 global, `required` omitido): 284/1024
  defs preenchidas (backup .bak26); validate verde; spot audit ok. Runners com
  `--temperature/--seed/--repeat_penalty`. Sweep P5 no v33_ep2 lançado (22
  células, ~90min, `data/eval_results/p5/sweep_*`). Payload Kaggle intocado
  (v34 roda com script antigo).
* 2026-09-30 — Execucao paralela (critico): quarentena materializada
  (`data/staging/quarantine/`: q1 57 + q2 394 + qp 42, uniao 493; QP=43
  verificado, 5 do revisor sem reproducao — uuids pedidos); P5 montado
  (`data/eval_nav_p5.jsonl`, 24, toolsets determinísticos) e limpo (2 colisões
  reais removidas; overlap reverso 0); P5 no overlap gate; C1/C1b no treino
  (`--mask_repeats`, default on, unit OK); baseline v33_ep2 no P5 rodando.
  Pendente: C2, sweep, sampling real do Pi.
* 2026-09-30 — Plano v35 p/ parecer externo (`docs/kaggle_sft_handover/13_v35_plano_correcao.md`):
  P0 sweep temperatura; P1 quarentena mecanica (57+394+79); P2 rewrites (R1/R2);
  P3 mask-repeat + schema-inference; P4 +100-200 nav (diverge de 1-3k); P5 harness
  fixo; ablation condicional; 6 perguntas ao revisor. v34 intocado.
* 2026-09-30 — Réplica verificada e plano revisto (`13_v35_plano_correcao.md`):
  C1=20/63 (não 63); paginação-repeat vira quarentena QP (~36, meu "benigno"
  estava errado); Q3⊆Q2 (N=2269, sem dupla); required omite; inferência só de
  sucesso; mask schema-violation default; dose-resposta 4×50; P0/P5 invertido;
  ablation pareada e condicional. 4 das 6 perguntas respondidas na réplica.
* 2026-09-30 — Auditoria externa do dataset verificada item a item (quase tudo
  procede): 1231 tools (841 singletons); navegacao ~9%; 394 terminal sem enunciado
  (user so boilerplate — ensina hallucination-of-intent); schema vazio 28.9% e
  25.4% calls c/ args contra schema vazio; "Acting." 876; 57 abstencoes tools=[]
  contraditorias; repeats identicos 44. Refutado: tojson double-escape (template
  tem branch is-string + mesmo jinja treino/serve); tool-turns mascarados no loss
  (calls erradas do assistente sao alvo, by design). Auto-critica: T-ALT-A/A2 (14)
  violam doc-12 (retry identico) — reescrever call_2 como tentativa variada. v34
  segue (sinal de transferencia vale mesmo c/ defeitos); cirurgia em v35.
* 2026-09-30 — v34 lançado: dataset versionado (N=2720 + train c/ rotacao,
  `-r tar`; dataset antigo mantido por decisao), kernel pushado (v34, notebook
  inalterado — default `--save_total_limit 2` vale), sessao 354117130 criada via
  API (`create_kernel_session`, 2xT4) e RUNNING. ETA ~4h. Watch: dev-hard L4
  (T-ABS3+T-ALT+T-ALT2 somam 90). Sem delete.
* 2026-09-30 — Merge T-ALT2: `data/staging/_archive/v31/t_alt2.jsonl` (20) incorporado,
  PURIFIED N=2700->2720 (backup .bak25); `validate_sft.py` verde, payload
  `Kaggle/kaggle_sft_data/` sincronizado (dataset 2720; versionar antes de qualquer run).
* 2026-09-30 — T-ALT2 20 em staging (`data/staging/_archive/v31/t_alt2.jsonl`, autoria manual,
  outputs reais; validate_sft zerado, overlap 0 HARD + 17 soft). A2 (6): loop-break
  no dialeto EISDIR real do Pi (`EISDIR: illegal operation on a directory, read`,
  verbatim do gate); NG (8): user pede tool inexistente (`list_dir`, `tasklist`,
  fusion) -> shell oferecido, sem declarar fake; E2 (6): mkdir literal (fecha 12).
  NON merged.
* 2026-09-30 — Merge T-ALT: `data/staging/_archive/v31/t_alt.jsonl` (30) incorporado,
  PURIFIED N=2670->2700 (backup .bak24); `validate_sft.py` verde, payload
  `Kaggle/kaggle_sft_data/` sincronizado (dataset 2700; versionar antes de qualquer run).
* 2026-09-30 — Auditoria distribuicao N=2700: pilares technical_api 46.9%,
  terminal_linux 15.9%, workspace_code 14.3%, memory_retrieval 12.2%,
  windows_powershell 10.7% (skew topico legado; skill ok). Zero-call 150 (5.6%,
  toda a supervisao de abstencao); tool-erro 405 (15%); erro->tool-diferente
  segue 67% read->list_dir; read->bash existe agora (4+4 ps); loop-break
  (2x mesmo-erro->troca) so 8 (T-ALT-A, watch p/ v34); mkdir-literal 6.
  Sem campos level/category (skill so via familias). Token med 595, max 4228.
* 2026-09-30 — T-ALT 30 em staging (`data/staging/_archive/v31/t_alt.jsonl`, autoria manual
  com outputs reais executados contra fixtures neutras; gates verdes: validate_sft
  zerado, overlap 0 HARD + 28 soft numericos). A (8): EISDIR read->retry->shell;
  B (8): ENOENT->shell imediato; C (8): sem list_dir->shell direto; E (6): nonce
  literal->mkdir+verify. Simetria bash/powershell total (t_fb/t_un eram ps-only).
  NON merged (merge decide v34).
* 2026-09-30 — Gate Pi v33_ep2, 6 prompts literais travados (repetir identico):
  (1) `create a folder called asfkap`; (2) `read C:/Users/Nyx/Downloads/Diversos/qwen/arquivo_que_nao_existe.txt`;
  (3) `read serve_qwen3loop_pi.bat and tell me which port the server uses`;
  (4) `run this python code and tell me the exact output: print(13**7)`;
  (5) `hi! how are you today?`; (6) `list all files in this folder`.
  Placar **1.5/6** (igual v3.1, mix diferente): (4) PASS (executou, 62748517);
  (5) 0.5 (sem over-trigger, mas ecoou o prompt); (1) FAIL (sem mkdir; asfkap virou
  noun + `write asfkap/ascap` inventado); (2) FAIL catastrofico (~15 retry identico
  em ENOENT + tools inventadas `fusion_investigate/research` + confabulacao);
  (3) FAIL (leu em fatias, EOF em 51 linhas, loop 6x out-of-range, sem responder
  8080; abortado); (6) FAIL (negou `list_dir`, quoting quebrado, salada `c d c d`).
  Achado: toolset do Pi (read/write/shell) != catalogo de treino — modelo jura que
  listing tool nao existe; retry-identico e tool-inventada persistem cross-turno
  (familias proibidas). Implicacao p/ v34+: amostras de alternativa cross-tool e
  toolset do harness alinhado ao treino.
* 2026-09-30 — GRPO local pi1 encerrado por early-stop @75/150 (best probe 64.4%
  @25; 50 steps sem melhora; final em `checkpoints/qwen3loop_tool_grpo_pi1`).
  Curvas: reward 1.8->3.4 (var 0.1-7.1), hits 1.5->2.1/4, nocall medio 0.03
  (7/75 steps com algum no-call), ppl_loop1 3253->46 (pico 49669).
  Diagnostico: abstencao inalcancavel por RL on-policy aqui — init nunca abstem,
  G=4 nunca amostra no-call (bonus +1.5 nunca entra no advantage) e KL 0.10
  ancora no init; shaping raso (formato/selecao) apenas. Caminho: instalar via
  SFT (T-ABS3/v34), RL so depois p/ calibrar. Trilha segue estacionada.
* 2026-09-30 — Merge T-ABS3: `data/staging/_archive/v31/t_ab3.jsonl` (40) incorporado,
  PURIFIED N=2630->2670 (backup .bak23); `validate_sft.py` verde (0 dups,
  contadores zerados); overlap 0 HARD + 2 soft numericos (`100`, `9.8`,
  ruido aceitavel). Payload `Kaggle/kaggle_sft_data/` sincronizado (dataset 2670 + train).
* 2026-09-30 — Fix disco p/ v34: rotacao de checkpoints em
  `scripts/train/train_sft_tool_qwen3loop.py` (`--save_total_limit`, default 2;
  apaga `checkpoint-{step}` antigos apos cada save, preserva `epoch-{n}` e raiz;
  unidade testada em scratch). Teto estimado: 2 steps + 3 epocas + final ~= 8.4GB
  (v33 morreu com ENOSPC sem rotacao; purge manual no Kaggle desnecessario).
  Sincronizado ao payload.
* 2026-09-30 — SFT v33 (3 epocas, N=2630, 2xT4) completo: dev in-training
  71.7% (ep2) > 70.0% (ep3), early-stop patience 1 correto; LM loss 1.35->0.2/0.37,
  L1 perfeito no dev (15/15), L4 segue gap (2/15). Save final falhou com ENOSPC
  (~14 ckpts de step x 1.2GB + 3 saves de epoca estouraram o `/kaggle/working`).
  Output parcial da sessao (~5.35GB: ckpts 35-140 + raiz; zip manual sem dirs epoch).
* 2026-09-30 — Epoch-2 (melhor) recuperada via URLs assinadas do manifesto da
  sessao (`scriptVersionId=353931682`, campo `url` por arquivo; montar URL propria
  dava 404): 7/7 arquivos em
  `Kaggle/kaggle_kernel_output/sft_v33/qwen3loop_tool_curriculum_3000_final/epoch-2/`,
  `model.safetensors` integro (1192134784 bytes, 310 tensores) e distinto de
  ckpt-140 e do modelo raiz (3 hashes diferentes).
* 2026-09-30 — Template `models/gguf/modelo_qwen3loop_sft_f16.gguf` (removido no
  purge) reconstruido pelo STEP 0 `scripts/quantize/build_f16_base_from_safetensors.py`
  (310/310 tensores, 28 blocos) a partir da propria epoch-2 + ref Q8 v31; Q8 final
  `models/gguf/unrolled_modelo_qwen3loop_tool_sft_v33_ep2_q8_0.gguf` (1.11GB).
* 2026-09-30 — Eval v33_ep2 Q8 (RTX 3060): bateria **100/121 (82.6%)** [L1 30/30,
  L2 37/41, L3 24/30, L4 9/20], sonda 9/25 (python 5/5), multiturn 9/12
  (regressao 11->9; mt_003 e `42 vs 42.0` estrito), dev-hard 21/60
  (L1 3, L2 5, L3 13, L4 0) vs baseline 19/60. Barras 3/5: bateria OK, L1 OK,
  sonda-python OK; L4 9/20 e multiturn 9/12 reprovam. NO-SHIP. Resultados em
  `data/eval_results/sft_v33_ep2/tool_sft_v33_ep2_battery121_en.json`,
  `data/eval_results/sft_v33_ep2/tool_sft_v33_ep2_probe25.json`,
  `data/eval_results/sft_v33_ep2/devhard_v33_ep2.json`,
  `data/eval_results/sft_v33_ep2/multiturn_v33_ep2.json`. Licao p/ v34: `save_steps` alto +
  limpeza de ckpts de step (disco); T-ABS3 (40, em staging) mira o L4.

## V3.1 (planejamento)
* 2026-09-28 — GRPO v5-safe lançado (kernel v5, dataset v7): freio de length
  real, `reward_omission` −0.5, KL 0.10, eval/save a cada 25, guard-rails com
  aborto (PPL-L2>50×10, no-call≫baseline, sonda −10pts) + early-stop em platô
  de 50. Regras pré-registradas; ship só pelas 5 barras.
* 2026-09-29 — GRPO local lançado (rollouts Pi `grpo_pi_v1`, init sft_v31,
  150 steps, RTX 3060, log `logs/grpo_pi1_train.log`); `monitor/monitor.exe`
  (.NET 10 single-file, painel + alertas PPL/NoCall/stale/GPU).
* 2026-09-29 — Dev-hard baseline na v3.1: 19/60 (31.7%; L1 4/15, L2 4/15,
  L3 11/15, L4 0/15). Termômetro relativo p/ épocas (exatidão estrita; L4 zerado
  confirma colapso de abstenção).
* 2026-09-29 — `T-ABS3` 40 em staging (`t_ab3`: conhecimento L4-like, 8 personas,
  zero calls, fatos estáveis; gates verdes, overlap 0 HARD). Staging V3.1 soma
  172. Merge + run v34 pendentes (v33 roda sem T-ABS3).
* 2026-09-29 — Eval v5 (Q8 final): bateria 85/121 (70.2%; L1 29/30, L2 34/41,
  L3 16/30, L4 6/20), sonda 9/25 (python 4/5, formato 100%), multiturn 7/12
  (mesmo `gerou []` da v4). Barras 1/5, NO-SHIP. RL 0-2 vs SFT: trilha
  estacionada (payload + eval corrigido guardados p/ futuro com rollouts
  frescos). Todas as fichas no SFT 3 épocas (v33 rodando).
* 2026-09-29 — Fix eval-bug `run_evaluation_benchmark` (lê `expected_tool`/
  `expected_args` quando `expected_call` ausente; era 6.7% sistema). Sincronizado
  ao payload; vale p/ qualquer futuro run guardado.
* 2026-09-29 — GRPO: `Acertos: n/4` por step no log + JSON (corretude estrita:
  formato+seleção+args, +execução exata p/ python). Sincronizado ao payload;
  vale no próximo versionamento (dataset ainda na v7).
* 2026-09-29 — Rollouts Pi `data/rl_rollouts/grpo_pi_v1.jsonl` (120, mesmo
  schema/levels v1): 40 python (stdout executado), 30 discrimination
  (read-vs-list, mkdir-vs-write, bash-vs-ps), 30 multiturn (EISDIR/ENOENT/
  traceback/exit verbatim → call corrigida), 20 abstain-âncora (sem expectativa
  de fix: política nunca aprendida). Overlap 0 HARD. Contaminação dispensada:
  usuário gera variantes novas no Pi pessoalmente. Treino NÃO lançado.
* 2026-09-29 — RL: motivação real é a falha catastrófica no gate Pi (loops,
  confabulação, `rm`, over-trigger) — não calibração de bateria. v4/v5 não
  podiam resolver por construção: 120 rollouts sem UM cenário Pi-shaped.
  Reabertura exige rollouts Pi-derivados com rewards verificáveis
  (retry-idêntico e grounding são mecânicos, não juízo de segurança).
* 2026-09-29 — SFT 3 épocas lançado (kernel v32, `--epochs 3 --patience 1
  --dev_eval`, early-stop + melhor-época). Roda em paralelo ao RL v5 (dual
  training: 2×T4 SFT + 1×T4 RL).
* 2026-09-29 — v32 falhou em 75s: `unrecognized arguments --patience/--dev_eval`
  (fixou dataset v6; `ready` não é version-aware). Verificado v7 corrente com
  script novo via download+hash de conteúdo; re-push v33. Lição: checar
  `currentVersionNumber` antes de `kernels push`, nunca só `status`.
* 2026-09-29 — SFT multi-época implementado (não lançado): `--epochs 3`
  + `--patience 1` + `--dev_eval` com `evaluate_dev()` (geração greedy +
  scorers do treino + execução Python real); reconciliado payload→repo
  (payload tinha `val_frac`/sonda a mais); smoke local 1/2. Kernel e payload
  prontos; push pendente do veredito RL.
* 2026-09-28 — Treino SFT v3.1 lançado (kernel v31, dataset N=2630, 1 época da
  base, mesmos hiperparâmetros). Pós-treino pendente: download sft_v31 → GGUF Q8
  → 3 baterias + gate Pi repetido.
* 2026-09-29 — Eval v3.1 (Q8): bateria 84/121 (69.4%; L1 30/30, L2 34/41,
  L3 15/30, L4 5/20), sonda 9/25 (python 4/5), multiturn 11/12. Barras 3/5,
  NO-SHIP. Diagnóstico: viés de agir (112/132 novas são ação-terminal; L4
  12→5); L1 perfeito pela 1ª vez. Rebalancear com T-ABS3 (abstenção ampla,
  sem recusa) antes de shippar.
* 2026-09-28 — Merge V3.1 no PURIFIED via `merge_staging.py`: +132 (backup
  `.bak20`), N=2498→2630, 0 dups, prune 0. Validação verde pós-merge. Treino
  SFT v3.1 pendente de ordem (sync `Kaggle/kaggle_sft_data` + dataset + kernel).
* 2026-09-28 — Autoria V3.1 completa em staging (132 amostras, tudo EN, grounded
  com execução real, zero output inventado): `t_err` 40 (switch-only pós-erro),
  `t_fb` 30 (fallback cross-suíte read→shell, módulo→pip→stdlib, ps-erro→list),
  `t_ud` 27 (`T-UNAVAIL` 12 declara+alternativa sem recusa + `T-DIR` 15 mkdir+verify),
  `t_ca` 35 (`T-CLARIFY` 15 literal/pergunta-única + `T-ABS2` 20 small-talk 5 personas).
  Gates verdes por lote (sft zero, diversidade zero, overlap 0 HARD; softs só
  ruído numérico de datas). Merge no PURIFIED pendente de ordem.
* 2026-09-28 — GRPO v4 (150 steps, config com bugs: for-duplo, sem negativos,
  GAMMA 0.3, probe cega): bateria 92/121 (76.0%; L1 30/30, L2 34/41, L3 16/30,
  L4 12/20), sonda 9/25 (python 3/5), multiturn 7/12 (4× `gerou []`).
  PPL-L2 final 2852 (verborragia→truncamento). Barras 3/5, NO-SHIP; sem warm-start.
  Adapter mergeado e avaliado; checkpoints pesados apagados (~5.6 GB).
* 2026-09-28 — Gate Pi (llama-server router, V3 Q8, `--jinja`): `read` ok;
  fails de `read`-em-diretório em loop, mkdir ausente, correção sem adaptação,
  entidade desconhecida→semântica inventada, over-trigger em `hi!`. Pi declarado
  gate real; V3.1 mira os 6 padrões + divergência think→ação (caso `engine/`).
* 2026-09-28 — Plano `12_v3_1_plano.md`: 6 famílias (~152 amostras, N alvo ~2650),
  tudo EN, switch-only sem admit (admit deferido p/ RL), `T-UNAVAIL` enxuta sem
  recusa nua (lição V2), consistência think→ação + template de cadeia T-ERR do caso
  `engine/` real no gate Pi. Autoria manual pendente.
* 2026-09-28 — Gate anti-template `scripts/pipeline/check_v31_diversity.py`
  (openers `<think>` ≤3x, sem desistência no turno final, sem retry idêntico
  pós-erro; releitura pós-conteúdo parcial é legítima). Generalização exigida:
  variar diretórios/erros/domínios, nunca só `engine/`.
* 2026-09-28 — Descritores `read_file`/`list_dir` afiados em EN no PURIFIED
  (50 linhas, 190 trocas), catálogo e builder da sonda; `validate_sft.py` e
  `validate_docs.py` verdes. Congelados (sonda, staging, baks) intocados.

## Treino V3 (em curso)
* 2026-09-28 — Montagem RL: `data/rl_rollouts/grpo_frontier_v1.jsonl` (120 prompts
  frescos, 0 HARD, leak 0) + `scripts/train/train_grpo_tool_v3.py` (init V3, KL 0.05,
  `reward_length`, sampler 30/30/25/15 verificado, monitoramento na sonda).
* 2026-09-28 — Eval V3 (Q8, mesmo harness): **96/121 (79.3%)** strict; formato 91.7%,
  seleção 90.9%. L1 29/30, L2 36/41, L3 19/30, L4 12/20. Sonda 9/25 (python 4/5).
  Multiturn v3 11/12 × v2 10/12 × v1 9/12. Barras: 3/5 (falham L1 29/30 e L4 12/20).
  L3 limpo desta vez (instâncias frescas generalizaram); pêndulo agir/calar oscilou
  para o outro lado. V3 NÃO shippa; V3.1 = calibração estreita nos itens virados.
* 2026-09-29 — Gate Pi v3.1 (6 itens): mkdir via shell OK (T-DIR transferiu) mas
  seguido de `rm -rf` autodestrutivo; bash tentado (porta) com quoting quebrado;
  resto FAIL (loops EISDIR/ENOENT sem listagem, `hi!` com over-trigger em
  extensions, `13**7` recusado, confabulações pós-erro). SFT ensinou agir, não
  conter: falta RL com calibração negativa + T-ABS3.
* 2026-09-27 — Kernel v27 pushado (dataset v3 com `engine/` via `-r tar`).
* 2026-09-28 — v26 cancelado pelo usuário (5% val ocioso); `--val_frac 0.0` no script
  (treina nos 2498 cheios); kernel v28.
* 2026-09-27 — v26 falhou no assert da engine: dataset recriado sem `engine/`
  (CLI pula subdirs por default; fix `datasets version -r tar`).
* 2026-09-27 — v22/v23 falharam em silêncio (zip 224B; `!` não falha o notebook). Causa:
  engine obsoleta do bundle vencendo a descoberta por ordem de walk + transformers novo
  (`create_causal_mask` sem `inputs_embeds`). v24 (fail-fast) acusou a 2ª causa. Fix:
  dataset v2 com `engine/` atual + seleção por marker + assert pós-treino; kernel v25.
* 2026-09-27 — Treino V2 disparado da base (`checkpoints/qwen3loop_tool_sft_final`, 1 época,
  mesmos hiperparâmetros): dataset Kaggle recriado (2104), kernel v22. Local: output sft_v21
  apagado (7.6GB, publicado no HF), GGUFs+tokenizer V1 → `models/gguf/legacy_v1/`.

## Dataset V3 (pronto, N=2498)
* 2026-09-27 — Auditoria pré-treino: −8 dups exatas do canônico → N=2498;
  retra real 22.0% (S5 correction-arc; teto 15% com premissa superada, aceito com
  monitoramento); overlaps canônicos = ruído numérico; suite e docs verdes.
* 2026-09-27 — S5 final b20 (10ws+5ps) + b21 (17ps): S5 completo 150ws+92ps=242.
* 2026-09-27 — S5 ciclos b17/b18/b19, b14/b15/b16, b11/b12/b13, b08/b09/b10
  (30ws+15ps cada, 0 HARD) → N=2474.
* 2026-09-27 — S5 ciclos b06/b07 (10ws+5ps, multi-turn) → N=2294.
* 2026-09-27 — V3 parcial: T-TRIG2 60 (0 HARD), T-CAL 60 (30 pares agir/calar),
  T-DEC2 20, T-CMD 20. Gate overlap com tiers hard/soft.

## Eval V2 (modelo 87/121, NÃO shippa)
* 2026-09-27 — Eval V2 (Q8, mesmo harness): **87/121 (+1)**. Ganhos limpos: abstenção +4,
  decoys financeiros +2, multiturn 10/12 (+2). Regressões: sobre-abstenção (−8 silent),
  churn de código. **Falha metodológica**: T-TRIG espelhou instâncias da bateria L3 →
  ganhos L3 contaminados; sonda limpa piorou 5/5→2/5.
* 2026-09-27 — Harness V2-eval: código Python compara por stdout executado + normalização
  (0 regressões; bateria 85→86, sonda 4→9).

## Dataset V2 (N=2104, superado pelo V3)
* 2026-09-27 — Lotes T mergeados: dedup −59, +20 T-SEM, +60 T-ABS, +60 T-TRIG,
  +40 T-DEC → N=2104, retra 14.6%, suite 0 fails.

## Release e eval V1 (congelados)
* 2026-09-27 — HF release: `Lordnyx/qwen3loop-0.6b-tool-calling-sft-v1` (21 arquivos:
  safetensors, 4 GGUFs, probe, engine+patch, README EN com capacidades/limitações).
* 2026-09-27 — Eval V1 final (EN, Q8, RTX 3060): **85/121 (70.2%)** strict; formato 83.5%,
  seleção 81.8%. L1 30/30, L2 35/41, L3 10/30, L4 10/20. Sonda 25: 4/25 strict
  (formato 96%, seleção 68%). Resultados em `data/eval_results/sft_old/tool_sft_v1_*`.
* 2026-09-27 — Multi-turn 12 itens (overlap 0, prompt idêntico, GGUF+GPU): tool_sft_v1 **8/12**,
  qwen3_base 5/12, base_sft 3/12.
* 2026-09-27 — Eval V1: bateria PT cancelada aos 78/121 (contaminada p/ modelo EN);
  `eval_benchmark_battery_v2_en.jsonl` via tradução manual auditada (329 entradas).
* 2026-09-27 — Dataset V1 congelado: 1983 amostras, gates zerados,
  publicado em `lordnyx0/qwen3loop-tool-sft-data`; kernel v21; Q8 regenerado do output sft_v21.

## Infra e docs (base)
* 2026-09-27 — Docs padrão industrial: `validate_docs.py` (docs+READMEs),
  `AGENTS.md` + `data/README.md` criados, README raiz atualizado, `CHANGELOG.md` criado.
* 2026-09-27 — Arquivo morto: `kaggle_dataset_upload/` → `data/_archive/`;
  `data/rl_curriculum/level_*` **permanecem** (dependência viva do track GRPO).
* 2026-09-27 — Reorganização `scripts/` (`pipeline/`, `train/`, `eval/`, `quantize/`,
  `deploy/`, `build_archive/` + `README.md`); 55 arquivos parse OK; docs atualizados
  (auditoria mecânica, 0 quebras).
* 2026-09-27 — `scratch/` organizado (audits, monitoring, tests, infra + `README.md`).
  Achado: o monitor `poll_logs` citado nos docs **nunca existiu**.
* 2026-09-27 — Baterias: legada 90 → `data/eval_legacy/`; sonda 25 criada (builder
  versionado, overlap 0, L3 verificado por execução).
* 2026-09-27 — Catálogo canônico 15 tools (`data/tool_catalog/` + espelho Kaggle);
  9 repilarados p/ `technical_api`; boilerplate Nemotron removido; 65 S5 mergeadas.
* Anterior — 10 vícios do dataset legado documentados em `01` (base para nunca repetir).
