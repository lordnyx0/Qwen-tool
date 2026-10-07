# Handover Técnico: Pipeline SFT Tool-Calling no Kaggle (Dual Tesla T4)

Este diretório contém o dossiê completo de transição e governança técnica para que qualquer novo agente ou engenheiro assuma, audite, monte o novo dataset e execute o treinamento supervisionado (Full-Parameter SFT) do modelo **Qwen3Loop-0.6B** na infraestrutura remota do Kaggle.

---

## 📌 Contexto Executivo e Estado Atual do Projeto

1. **Arquitetura Base:**
   * **Modelo:** Qwen3Loop-0.6B (594M parâmetros físicos em 28 camadas físicas, com tied recurrence unrolling para 56 camadas lógicas em inferência/GGUF).
   * **Objetivo do Treino:** Full-FT SFT focado 100% em **Tool Calling Estrito**, resolução de problemas no terminal (`bash` e `powershell`), chamadas de APIs e auto-correção de trajetórias (retrajetórias).
   * **Status da Máquina Local:** 0% de uso de GPU local (RTX 3060). Todo o treinamento pesado roda remotamente no cluster Kaggle (2x NVIDIA Tesla T4, 16 GB VRAM cada).
   * **Espaço Local:** Limpo e purgado (reduzido de 36.15 GB para 7.79 GB; 28.36 GB de artefatos obsoletos e duplicatas foram removidos com sucesso).

2. **Cancelamento do Run Anterior (Veredito Crítico):**
   * O dataset prévio (`curriculum_3000_tool_sft.jsonl`) foi cancelado antes do término do treino devido a **10 patologias críticas de dados** (esquizofrenia de formatos string vs. OpenAI struct, payloads maliciosos não filtrados do PSStrikes, observações fake/vácuas, tipos inválidos `"dict"`, vocabulário esparso de 1.799 tools e duplicações sintéticas).
   * **Decisão:** Reconstruir o dataset a partir das fontes puras sob um catálogo canônico denso (< 50 ferramentas), sem dependências tóxicas.

---

## 🗂️ Índice dos Documentos de Handover

| Documento | Descrição | Estado em 2026-09-27 |
| :--- | :--- | :--- |
| 📋 [**01_diagnostico_dataset_legado.md**](01_diagnostico_dataset_legado.md) | **Post-Mortem Técnico:** Análise detalhada dos 10 vícios críticos que inviabilizaram o dataset anterior e o que JAMAIS deve ser repetido. | Vigente (+nota single-turn) |
| 🎯 [**02_diretrizes_curadoria_dataset.md**](02_diretrizes_curadoria_dataset.md) | **Especificação do Novo Dataset:** Padrão ChatML canônico, catálogo fechado de < 50 tools densas, sanitização de segurança (bloqueio de malware), eliminação de PII e observações substantivas. | Vigente |
| ⚡ [**03_pipeline_kaggle_e_infraestrutura.md**](03_pipeline_kaggle_e_infraestrutura.md) | **Guia de Deploy no Kaggle:** Dual GPU DDP (Accelerate), BitsAndBytes 8-bit AdamW, Gradient Checkpointing (`use_reentrant=False`), SDP Cutlass Turing, empacotamento `-r tar` e comandos da CLI do Kaggle. | Vigente (comandos verificados) |
| 🔄 [**05_metodo_treino_e_loop_recorrente.md**](05_metodo_treino_e_loop_recorrente.md) | **Método de Treino e Recorrência (Loop):** Formulação da Multi-Exit Loss ($w_1=0.4, w_2=1.0$), amarração de pesos (tied recurrence), KV cache virtual (56 camadas), máscara `assistant_only_loss`, sonda latente e unrolling GGUF. | Vigente |
| 🌐 [**06_fontes_datasets_e_scripts.md**](06_fontes_datasets_e_scripts.md) | **Repositórios e Scripts:** Links diretos no Hugging Face (Nemotron, Hermes, PSStrikes, OpenClaw), IDs no Kaggle e inventário completo de scripts de extração, treino, deploy e avaliação. | Vigente (paths atualizados) |
| 🚀 [**04_checklist_e_proximos_passos.md**](04_checklist_e_proximos_passos.md) | **Roteiro de Ação Imediata:** Checklist sequencial com comandos prontos para o novo agente gerar os dados limpos, subir para o Kaggle, treinar, baixar e quantizar em GGUF Q8_0. | Parcialmente executado (Passos 1-4 em curso) |
| ⚖️ [**07_singleturn_vs_midtrajectory.md**](07_singleturn_vs_midtrajectory.md) | **Correção de Critério:** single-turn `user→assistant(tool_calls)` é predição válida (manter); só órfão mid-trajectory trunca; `tool_response` sintetizado é proibido (anti-padrão). | Vigente |
| 📦 [**08_excecoes_aceitas.md**](08_excecoes_aceitas.md) | **Dívidas Documentadas:** technical_api diversa (grav. 2), think verboso (grav. 4), E3 repilarada e executada. | Vigente |
| 🏁 [**V1_MANIFEST.md**](V1_MANIFEST.md) | **Dataset V1 Congelado:** 1983 amostras, gates zerados, payload publicado, kernel v21 disparado. | Congelado (imutável) |
| 🗺️ [**09_v2_roadmap.md**](09_v2_roadmap.md) | **Plano V2:** fila S5 (ws→200, ps→250), merge canônico, benchmark holdout. | Em execução |
| ⚖️ [**07_singleturn_vs_midtrajectory.md**](07_singleturn_vs_midtrajectory.md) | **Correção de Critério:** single-turn `user→assistant(tool_calls)` é predição válida (manter); só órfão mid-trajectory trunca; `tool_response` sintetizado é proibido (anti-padrão). |
| 📦 [**08_excecoes_aceitas.md**](08_excecoes_aceitas.md) | **Dívidas Documentadas:** technical_api diversa (grav. 2), think verboso (grav. 4), E3 repilarada e executada. |
| 🏁 [**V1_MANIFEST.md**](V1_MANIFEST.md) | **Dataset V1 Congelado:** 1983 amostras, gates zerados, payload publicado, kernel v21 disparado. |
| 🗺️ [**09_v2_roadmap.md**](09_v2_roadmap.md) | **Plano V2:** fila S5 (ws→200, ps→250), merge canônico, benchmark holdout. |
| 📂 `../data/staging/` | **Staging S5 + exemplares** (dados saíram desta pasta de docs; ver layout em `data/`). |

---

## 🛡️ Regras Inegociáveis para o Novo Agente

1. **Zero Treinamento Local:** Nunca inicie o treinamento de época completa na máquina local do usuário. Use os nós Dual Tesla T4 no Kaggle via script de deploy.
2. **Formato ChatML Nativo Único:** Todo o dataset deve seguir estritamente o formato de `tools` no topo, `tool_calls` estruturados em turnos de `assistant`, e `tool` como papel de resposta, com JSON estritamente parseável (`json.loads`).
3. **Segurança de Código:** Payloads de exploit ofensivo (Mimikatz, DumpWCM, evasão de Defender) são estritamente proibidos no SFT. O foco é administração legítima e engenharia de sistemas.
4. **Sem Observações Fake:** Se o assistente emite uma chamada, a resposta da ferramenta deve simular ou conter o output real da ferramenta (não textos vazios ou placeholders como `"Command completed successfully with exit code 0"`).
