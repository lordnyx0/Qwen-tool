COMO AUTORAR (manual; scripts nao sintetizam conteudo - regra do repo)
================================================================
Alvo: itens 2-4 do plano especialista (irrelevancia c/ catalogo, missing-arg,
structured, multiturn). Meta: 150-300 por categoria. EN only (decisao 2026-10-01).

Arquivos aqui:
- *_shell.jsonl .... concha vazia com as chaves obrigatorias (uuid/source/tools/messages).
  Troque TEMPLATE_* por conteudo real. NUNCA funda a concha (merge aborta? nao -
  ela passaria; apenas nao a inclua: o validador 'generica' a barraria, mas nem envie).
- candidates_structured.jsonl ... 120 amostras reais com schema aninhado (base p/ item 3).
- candidates_multiturn.txt ..... 193 tools com 2-12 amostras (pares p/ continuacoes, item 4).

Regras de conteudo (validadas por scripts/pipeline/validate_sft.py):
1. outputs de tool DEVEM ser executados de verdade (verbo do fixture/log), nunca inventados.
2. toda call referencia tool declarada; descricoes citam SO tools declaradas (regra desc_decl:
   lista fechada list_dir/read_file/bash/powershell/write_file + aliases list_files/cat_file/
   shell/pshell/save_file).
3. sem PII real; sem comandos ofensivos (blocklist em 02_diretrizes_curadoria_dataset.md).
4. irrelevancia: catalogo presente (5-20 tools) e resposta final SEM call, explicando
   por que nenhuma serve. missing_arg: usuario omite obrigatorio -> assistant PERGUNTA
   (termina em '?'), sem chamar.
5. uuid novo unico, source manual_v37. Fundir SO via scripts/pipeline/merge_staging.py
   apos validate + check_eval_overlap (0 HARD) no arquivo staging.

HARNESS LOCAL (execucao real p/ itens 3-4; scripts/pipeline/harness_local.py)
================================================================
TOOLS: read_file(path) / list_dir(path) / powershell(command) /
  execute_python(code). Jail: fixture_fs/ (sem '..' p/ fora), blocklist de
  comandos destrutivos/rede, timeout 20s, uso manual local.
  list_dir retorna entradas ordenadas (contrato do harness).
FIXTURES: fixture_fs/ (15 arquivos, mtime 2026-01-01, inclui binario,
  dir vazio, erros reais) + fixtures_exec.jsonl (29 pares call->output
  verbatim: 22 ok + 7 erros/BLOCKs). Re-gerar: build_fixture_fs.py, depois
  harness_local.py --calls <seeds> --out <fixtures>.
GATE EXTRA p/ lotes novos: validate_calls_schema.py --gate (calls vs defs).
  Legado tem 34 violacoes de tipo em 25 purified (quarentena/qp7, reversivel).
