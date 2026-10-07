# AGENTS.md — convenções para agentes neste repo

Leia antes de agir: `docs/kaggle_sft_handover/README.md` (índice), `09_v2_roadmap.md` (fila atual),
`V1_MANIFEST.md` (congelado) + `data/README.md` + `scripts/README.md` + `scratch/README.md`.

## Regras de operação
1. **CWD = raiz do repo** em todo comando. Rode scripts como `python scripts/<pasta>/<arq>.py`.
2. **Single-writer** em `data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl`: nunca edite em paralelo
   (nem 2 agentes, nem agente + humano). Staging entra só via `scripts/pipeline/merge_staging.py`.
3. **Backup antes de escrita em massa**: `.bakN` numerado do PURIFIED antes de qualquer merge/replaceAll.
4. **Sem one-liners engenhosos**: lógica Python vai em arquivo (`/Temp/opencode` p/ rascunho,
   `scripts/` se virar canônico). Inline `python -c` só para leitura/inspeção.
5. **Validação é gate, não enfeite**: `scripts/pipeline/validate_sft.py` (dados) e
   `scripts/pipeline/validate_docs.py` (docs) após cada lote. Promessa de subagente ≠ entrega.
6. **Mecânico liberado com cerca**: renames/dedup/sync/diffs determinísticos OK com dry-run + validação.
   **Redação de amostras liberada com trava** (regra anti-síntese derrubada pelo dono em 2026-10-06):
   agente pode redigir turnos user/assistant DESDE QUE observations sejam outputs reais
   verbatim (goldens) + variedade medida (users únicos, reuso de think capado, catálogos variados).
   Segue proibido: inventar outputs de execução, decidir segurança, classificar pilares.
7. **Subagentes**: recebem prompts fechados (formato exato + lista de domínios proibidos + regra sem-bash);
   escrevem SÓ em staging separados; merge só após validação.
8. **Docs**: todo path em code span ou `file://` deve existir (rode `validate_docs.py`); menções a
   arquivos inexistentes sem path; `CHANGELOG.md` do handover atualizado a cada mudança estrutural.
9. **Nunca**: treino local de época completa, exposição de PII real, comandos ofensivos (blocklist em
   `02_diretrizes_curadoria_dataset.md`), commit de `.bak`/caches, edições concorrentes no mesmo arquivo.
