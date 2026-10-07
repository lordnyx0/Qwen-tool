# 01. Post-Mortem e Diagnóstico do Dataset Legado

Este documento registra o diagnóstico detalhado e a autópsia técnica do arquivo `curriculum_3000_tool_sft.jsonl` (3.000 amostras, 16.6 MB), cuja execução foi cancelada por conter inconsistências graves de formato, qualidade e segurança.

---

## 🛑 Os 10 Vícios Críticos Identificados

### 1. Esquizofrenia de Formatos (String vs. Struct)
* **Problema:** O dataset misturou dois paradigmas incompatíveis:
  * **Hermes (500 amostras):** `tools: []`, `tool_calls: None`, com as declarações de ferramentas jogadas dentro do `system` prompt em `<tools>` e as chamadas emitidas como texto cru em `content` contendo `<tool_call>{...}</tool_call>`.
  * **Restante (2.500 amostras):** Formato nativo OpenAI/Qwen com a lista `tools` no cabeçalho do payload, o array de objetos estruturados `tool_calls: [{"id": "...", "type": "function", "function": {"name": "...", "arguments": "{...}"}}]` e o papel `role: "tool"`.
* **Impacto:** O modelo tenta aprender simultaneamente duas linguagens de saída: ora emitir tags textuais no meio do fluxo, ora preencher a estrutura formal de chamadas. Isso gera alucinações e quebras de tokenização no template Jinja (`chat_template.jinja`).

### 2. Tool-Calls Corrompidos e Ferramentas Não Declaradas
* **Hermes:** 483 de 974 chamadas possuíam JSONs inválidos que falham no `json.loads` (uso de aspas simples `{'queries': ['...']}`, chaves invertidas ou caracteres não escapados).
* **OpenClaw/Bellfire:** 345 de 475 amostras chamavam ferramentas como `browser`, `terminal`, `write_file` e `search_files`, mas o script apenas declarava `safe-exec`, `filesystem-context`, `memory_get` e `browser_open`. O modelo aprendia a chamar ferramentas inexistentes no catálogo.

### 3. Observações Vácuas e "Fake Outputs"
* **PSStrikes:** 350 amostras continham como retorno da ferramenta a frase idêntica e estéril:
  `"Command completed successfully with exit code 0."`
* **OpenClaw:** Centenas de respostas resumidas a `{"status": "ok"}` ou URLs estáticas (`https://example.com`).
* **ToolMind:** 301 de 438 turnos com chamadas órfãs sem a observação seguinte da ferramenta.
* **Impacto:** Treina o modelo a ignorar o output da ferramenta e impede o aprendizado de raciocínio reativo (inspecionar o retorno para tomar a próxima decisão).
* **Correção de critério (sessão manual, ver `07_singleturn_vs_midtrajectory.md`):** single-turn `user → assistant(tool_calls)` sem `tool` é formato válido de predição (Hermes/Glaive) e foi MANTIDO; só órfão *mid-trajectory* e *tool stray* foram removidos. Conversão com `tool_response` sintetizado é proibida (anti-padrão documentado).

### 4. Zero-Calls (Amostras Sem Chamada Alguma)
* **ToolMind:** 159 de 500 amostras (31.8%) não realizavam nenhuma chamada de ferramenta (`tool_calls` vazio), sendo apenas conversas textuais comuns (ex: assistente pedindo clarificação sobre `website_id`).
* **Synth Memory:** 156 de 500 amostras (31.2%) sem chamada de ferramenta.
* **Impacto:** Dilui o objetivo do fine-tuning focado em tool calling com conversação genérica já dominada pelo modelo base.

### 5. Schema JSON Inválido (`type: dict`)
* 632 amostras continham `"type": "dict"` nos esquemas dos parâmetros em vez do padrão OpenAPI / JSON Schema obrigatório `"type": "object"`. Isso quebra validadores rígidos e engines de inferência como vLLM e Ollama.

### 6. Vocabulário Hiper-Esparso (1.799 Ferramentas Distintas)
* A presença de 1.799 nomes de ferramentas únicas (a esmagadora maioria com apenas 1 exemplo no dataset) impede a generalização semântica. O modelo memoriza ruído em vez de dominar a relação entre a descrição da ferramenta e a geração correta dos parâmetros.

### 7. Duplicações Exatas Artificiais
* 171 amostras eram cópias redundantes de loops sintéticos:
  * `windows_powershell_admin`: 5 tarefas multiplicadas por 18 vezes (`* 18`).
  * `windows_powershell_recovery`: 5 tarefas multiplicadas por 12 vezes (`* 12`).
  * `openclaw_rec_*`: 5 tarefas multiplicadas por 5 vezes (`* 5`).
  * Nemotron continha 495 prompts idênticos repetindo o texto introdutório do Terminus.
  * OpenClaw continha 53 chamadas idênticas para `Read SOUL.md`.

### 8. Código Inválido e Placeholders Não Executáveis
* Blocos de código em Python que falham em `ast.parse`.
* Comandos PowerShell contendo templates não preenchidos (`${host.file.path}`, `<target_system>`, `Copy-Item ${...}`).

### 9. Bloqueador de Segurança: Malware e Exploits no PSStrikes
* O dataset `dessertlab/PSStrikes` foi concebido originalmente para red teaming e embutia payloads ofensivos reais sem nenhuma recusa ética ou defensiva:
  * Injeção de credenciais: `DumpWCM.ps1`, `iex(new-object net.webclient).downloadstring(...)`.
  * Extração de senhas com `Invoke-Mimikatz`.
  * Evasão de segurança: `-WindowStyle Hidden -ExecutionPolicy Bypass`, desativação de Defender e bypass de AMSI.
  * Phishing e exploração Kerberos (`Get Kerberoastable`).
* **Veredito:** Inadmissível para um modelo de engenharia de software e automação legítima.

### 10. Violação de Privacidade (PII) e Desbalanceamento Bimodal
* **PII:** Amostras em `synth_memory` persistiam informações identificáveis de menores de idade reais ou quasi-reais (ex: aluna de 16 anos com nome completo, escola em Austin e detalhes pessoais).
* **Desbalanceamento Bimodal de Tokens:**
  * OpenClaw: Média de 54 tokens por amostra.
  * PowerShell: Média de 98 tokens por amostra.
  * Hermes e Nemotron: Médias de 2.123 a 2.237 tokens por amostra.
  * O gradiente do treinamento sofria um viés bimodal extremo (ou passos microscópicos de 50 tokens, ou sequências pesadas de 4.000 tokens).

---

## 🎯 Conclusão e Diretriz para o Futuro
Qualquer nova compilação de dataset deve passar por testes automatizados estritos (AST, schema, detecção de regex ofensivo e paridade com o ChatML nativo do Qwen) antes de ser enviada ao Kaggle.
