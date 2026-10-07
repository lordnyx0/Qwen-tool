# 12 — V3.1: calibração SFT pós-gate Pi (plano)

Data: 2026-09-28. Status: planejamento; autoria manual pendente.
Base congelada: V3 (`data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl`, N=2498).
Protocolo de treino (inalterado): 1 dataset ↔ 1 modelo, sempre da base
`checkpoints/qwen3loop_tool_sft_final`, 1 época, lr 1.5e-5. Sem chaining V3→V3.1.

## 1. Diagnóstico que motiva (gate Pi, modelo V3)

1. Mau uso de `read` em diretório (EISDIR) em loop — pista de erro ignorada.
2. Recovery só intra-suíte (`read`→`list_dir` 53/60); fallback cross-tool (`read`→`bash`) 0/2498.
3. `write` default para "criar pasta"; correção do usuário sem mudança de estratégia.
4. Entidade desconhecida vira semântica inventada (`asfkap`→"AsfK algorithm").
5. Over-trigger em cumprimento (`hi!`→`read`+`write` não solicitados).
6. Afirmação de sucesso sem output de tool ("now successful").
7. Divergência think→ação (gate 2026-09-28, `engine/`): texto diz "Let's list it",
   `<tool_call>` emitido é `read` no diretório. Raw `<tool_call>` vazou como texto
   (parse Jinja/servidor — lado harness, nosso dado já tem 0 literais).

Descritores `read_file`/`list_dir` já afiados em EN no dataset, catálogo
(`data/tool_catalog/tools_schema.py`) e builder da sonda (50 linhas, 190 trocas,
gates verdes). O resto é dado novo.

## 2. Regras de autoria (juízo humano; agente não sintetiza conteúdo)

- Tudo em inglês (prompts, thinks, outputs). Zero PT.
- Formato exato: espelhar `s5_ws_0001` (system EN + user + assistant `<think>` +
  `tool_calls` estruturado + turnos `tool` + resumo final). Campo `tools` por amostra
  com os descritores novos em EN. `uuid` único por amostra.
- Outputs de tool DEVEM ser reais: rodar o comando de verdade e colar stdout
  (bash/python) ou listar diretório real (list_dir). Nada inventado.
- Sem PII/paths locais (`C:\Users\...` proibido; usar paths neutros `logs/`, `C:\Logs`).
- **Proibido em todas as famílias:** retry idêntico após erro; sucesso confabulado;
  `admitir e parar` (sem finais de desistência no SFT); recusa nua (bare refusal);
  think que nomeia uma tool e call que emite outra (consistência think→ação obrigatória).
- Template de cadeia `T-ERR` (caso `engine/` real): palpite de arquivo → ENOENT →
  `list_dir` no pai → `read_file` no arquivo encontrado → resumo. Sempre terminar
  no arquivo certo lido. Template é ESTRUTURA, não texto: variar diretórios, tipos
  de arquivo, erros (ENOENT, EISDIR, permissão, traceback python, exit≠0 em bash,
  memória vazia→ampliar busca) e domínios (não só arquivos).
- **Generalização anti-template (obrigatório):** nenhum opener de `<think>` pode se
  repetir em mais de 3 amostras (ex.: "Attempting to read..." está banido por
  saturação no S5); cada amostra formula o raciocínio com palavras próprias.
  Gate: `scripts/pipeline/check_v31_diversity.py` no staging (openers repetidos,
  frases de desistência no turno final, retry idêntico).
- Staging separado por família em `data/staging/v31/*.jsonl`; merge só via
  `scripts/pipeline/merge_staging.py` após validação. Single-writer no PURIFIED;
  backup `.bakN` antes de qualquer merge.
- Gates por lote: `scripts/pipeline/validate_sft.py` (zero rejeitos),
  `scripts/pipeline/check_eval_overlap.py` (0 HARD contra
  `data/eval_benchmark_battery_v2_en.jsonl`, `data/eval_probe_canonical_25.jsonl`,
  `data/eval_multiturn_v1.jsonl`), dedup exato.

## 3. Famílias (total ~152; N alvo ~2650 pré-dedup)

| ID | N | Padrão obrigatório | Terminal |
|---|---|---|---|
| `T-ERR` | 40 | erro de tool → **troca de tool** (switch-only) | sucesso verificado via tool diferente |
| `T-FALLBACK` | 30 | recovery cruzando suítes: read-erro→`bash ls`, list-erro→`bash`, bash-erro→`powershell`, write-erro→bash heredoc | sucesso verificável |
| `T-UNAVAIL` | 12 | tool necessária ausente → **declara + tenta alternativa na mesma trajetória** (nunca recusa nua; lição da V2 que se recusava demais) | sucesso via alternativa |
| `T-DIR` | 15 | criar diretório via `bash mkdir` + verificar via `list_dir` | listing confirma |
| `T-CLARIFY` | 15 | entidade desconhecida → agir literalmente; perguntar SÓ se alternativas forem destrutivas/ambíguas (perguntar não é recusar) | ação literal ou pergunta única |
| `T-ABS2` | 20 | cumprimento/small-talk com system prompts variados (personas estrangeiras tipo Pi) → resposta direta | zero tool calls |
| `T-ABS3` | 40 | conhecimento geral no formato da L4 (responder o que sabe, 8 personas) | zero tool calls, fatos estáveis |
| higiene | — | amostras novas usam `python_interpreter` (nome canônico; treino atual só viu `execute_python`) | — |

## 4. Deferido para RL (NÃO entra na V3.1)

- Comportamento "admitir e parar": calibrar via reward depois, nunca como demo SFT
  (risco de over-refusal estilo V2).
- Freio de verbosidade (`reward_length`): observar na v5 antes de mexer.

## 5. Sequência

1. Autoria manual por família → staging `data/staging/v31/`.
2. Validação por lote (sft + overlap + dedup) → `merge_staging.py` → N alvo.
3. Treino SFT v3.1 da base → GGUF Q8 → 3 baterias vs 5 barras (mesmo protocolo V3/v4).
4. Gate Pi repetido nos itens que falharam (EISDIR, mkdir, correção, `hi!`).
