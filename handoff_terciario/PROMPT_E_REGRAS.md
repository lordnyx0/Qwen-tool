# Agente terciário — curadoria generic vs narrow (2 arquivos, sem acesso externo)

Você tem acesso SOMENTE a esta pasta, com 2 arquivos: este prompt+regras e dataset_manifest.jsonl (3981 linhas, 1 por amostra, com uuid, executor, n_calls, n_tools, token_len, source, is_retrajectory, pattern). Sem shell, sem python, sem rede, sem repo. Só leitura e relatório em texto.

Divisão: coluna executor com valores narrow, generic, mixed, abstention. Filtre por ela. mixed (multi) + abstention (0 calls) são compartilhados fora dos tracks.

## Tarefa

Analise dataset_manifest.jsonl por executor x n_calls x source x token_len x pattern x retra.

Devolva Markdown nesta ordem:
1. COBERTURA: counts por executor, por n_calls, por source, por faixa de token_len, % retra. Aponte o slice mais fraco de generic vs narrow.
2. RISCOS: top 5 slices problemáticos com uuids reais do manifesto (template, catálogo pobre, single excessivo, retra acima do teto, source enviesada).
3. PROPOSTA: até 3 lotes novos (nome, filtro por executor/calls/token/source, n alvo, 1 user-exemplo marcado RASCUNHO NÃO-MERGEÁVEL no ChatML abaixo).
4. GATES: o que o primário deve rodar após sua proposta (validate_sft, backup .bakN, merge_staging, re-sync Kaggle).

## Regras normativas integrais (não resumidas)

### Formato ChatML obrigatório por amostra
Campos: uuid, tools no topo, messages, source, pillar, token_length, is_retrajectory.
Sequência: system, user (instrução), assistant com tool_calls, tool com mesmo tool_call_id e conteúdo substantivo real, assistant final conclusivo.
- arguments é string JSON parseável via json.dumps. Nunca dict bruto.
- Toda tool chamada deve estar declarada em tools.
- Cada tool_calls exige turno tool imediato. Proibido respostas genéricas ok, true, exit code 0 sem conteúdo técnico.
- 100% das amostras de Tool SFT com ao menos 1 call, exceto lotes de abstenção explícita.
- Termina em assistant. Blocos python passam em ast.parse.

### Catálogo canônico fechado
Terminal e shell: bash(command), powershell(command) — administração legítima, diagnósticos Test-NetConnection e Resolve-DnsName, serviços Get-Service e Restart-Service, auditoria de logs, DevOps.
Workspace: read_file(path, offset, limit), write_file(path, content), list_dir(path, recursive), search_code(query, path), execute_command(command).
Memória: core_memory_append(key, value), core_memory_replace(key, value), archival_memory_insert(content), archival_memory_search(query).
Técnica: python_interpreter(code), calculator(expression), http_request(url, method, headers, body), database_query(query).

### Segurança e PII (bloqueadores estritos)
Rejeitar qualquer comando com: mimikatz, dumpwcm, kerberoast, bypass, hidden, executionpolicy bypass, invoke-expression, iex(, downloadstring, c2, payload, meterpreter, defender bypass, amsi bypass, reverse shell.
Nenhum dado real de pessoas, escolas ou menores. Memória vira cenário enterprise abstrato.

### Balanceamento
Teto absoluto 5000 tokens. Alvo: curto 100-800 30%, médio 800-2000 50%, longo 2000-4500 20%. Retrajectories com erro real + correção: mínimo 10-15%, teto 16%.

### Single-turn vs órfão (norma 07)
Single-turn válido, MANTER: user para assistant(tool_calls) terminando aí, sem tool após. É predição P(call|user), padrão Hermes/Glaive.
Órfão mid-trajectory, DEFEITO: assistant(tool_calls) no meio seguido de user ou outro assistant em vez de tool. Tratamento: remover turno órfão final, manter prefixo, nunca inventar observation.
Tool stray, DEFEITO: tool sem tool_calls pendente. Remover o tool, manter resto.
Anti-padrão: tool_response sintetizado (ex. price inventado). Proibido.

### Exceções aceitas (08)
E1 technical_api diversa, gravidade 2, ACEITA (cada amostra carrega schema, sinal é tipagem estrita). E2 think Nemotron verboso, gravidade 4, ACEITO por enquanto. E3 9 amostras repilaradas para technical_api, EXECUTADA. E9 stdout bash vazio é legítimo (mkdir/cd/touch), ACEITO com escopo; para narrow read_file vazio é suspeito.

### Congelado e fila
V1 congelado N=1983, imutável. Fila V2: S5 workspace 50 para 200, powershell 158 para 250, priorizar bucket médio e retra:false. Benchmark holdout 90 só após fechar volume.

## Proibições do subagente
Não decidir segurança, não classificar pilares, não inventar outputs de execução (observations só goldens reais verbatim), não treinar época completa local, não expor PII, não usar comandos da blocklist. Não escrever no PURIFIED, V1 ou eval_results. Variedade obrigatória em exemplos: users únicos, think capado, catálogos variados. Entrega: só o relatório.
