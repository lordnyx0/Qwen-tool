# 02. Diretrizes para Curadoria e Especificação do Novo Dataset

Para garantir que o modelo aprenda generalização robusta e emita chamadas de ferramentas de forma precisa e segura, o novo dataset de SFT deve seguir as regras estruturais e de curadoria descritas neste guia.

---

## 🏗️ 1. Padrão ChatML Unificado e Obrigatório

Todas as amostras do arquivo `.jsonl` devem aderir rigidamente ao formato:

```json
{
  "uuid": "curated_tool_0001",
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "nome_da_ferramenta",
        "description": "Explicação concisa e precisa da utilidade da ferramenta.",
        "parameters": {
          "type": "object",
          "properties": {
            "param1": {
              "type": "string",
              "description": "Descrição do parâmetro."
            }
          },
          "required": ["param1"]
        }
      }
    }
  ],
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant equipped with specialized tools..."
    },
    {
      "role": "user",
      "content": "Instrução ou comando do usuário solicitando uma tarefa."
    },
    {
      "role": "assistant",
      "content": "<think>\nRaciocínio estratégico deliberativo (se aplicável).\n</think>",
      "tool_calls": [
        {
          "id": "call_001",
          "type": "function",
          "function": {
            "name": "nome_da_ferramenta",
            "arguments": "{\"param1\": \"valor_estrito\"}"
          }
        }
      ]
    },
    {
      "role": "tool",
      "name": "nome_da_ferramenta",
      "tool_call_id": "call_001",
      "content": "Output substantivo e realista gerado pela execução real ou simulada da ferramenta."
    },
    {
      "role": "assistant",
      "content": "Interpretação conclusiva do resultado obtido e entrega final ao usuário."
    }
  ],
  "source": "origem_limpa",
  "pillar": "terminal_linux | windows_powershell | workspace_code | memory_retrieval | technical_api",
  "token_length": 1250,
  "is_retrajectory": true
}
```

### Regras Estritas de Validação de ChatML:
1. **`arguments` deve ser JSON String parseável:** O valor do campo `arguments` deve ser uma string serializada via `json.dumps({...})`. NUNCA use dicionários Python brutos dentro de `arguments`.
2. **Declaração Obrigatória:** Toda ferramenta chamada em `tool_calls` DEVE obrigatoriamente estar descrita na lista `tools`.
3. **Observação Substantiva:** Cada `tool_calls` deve ser sucedido imediatamente pelo seu turno `role: "tool"`, com o mesmo `tool_call_id`. Respostas genéricas como `"ok"`, `"true"` ou `"exit code 0"` sem conteúdo técnico são proibidas.
4. **Sem Amostras "Zero-Call":** No dataset de Tool SFT, 100% das amostras devem conter ao menos uma chamada de ferramenta executada.

---

## 🛠️ 2. Catálogo Canônico Fechado (< 50 Ferramentas Densas)

Em vez de 1.799 ferramentas fragmentadas, as fontes devem ser consolidadas nas seguintes suítes canônicas densas:

### Suíte 1: Terminal & Shell (`terminal_suite`)
* `bash(command: string)`: Execução em ambiente Linux (navegação, compilação, `grep`, `sed`, `awk`, inspeção de processos).
* `powershell(command: string)`: Automação e administração legítima de sistemas Windows (`Get-Process`, `Get-Service`, `Get-ChildItem -Force`, `Select-String`, `Test-NetConnection`, manipulação de registro e scripts).

### Suíte 2: Workspace & Engenharia de Código (`workspace_suite`, MCP canônico desde 2026-10-07)
* `read_text_file(path: string, head?: integer, tail?: integer, offset?: integer)`: Leitura de arquivos locais.
* `read_media_file(path: string)`: Tipo e tamanho de mídia/binário.
* `read_multiple_files(paths: string[])`: Leitura concatenada de vários arquivos.
* `list_directory(path: string)`: Listagem estruturada de diretórios.
* `list_directory_with_sizes(path: string)`: Listagem com tamanhos.
* `directory_tree(path: string)`: Árvore indentada do diretório.
* `search_files(pattern: string, path?: string)`: Busca literal no código.
* `get_file_info(path: string)`: Metadados (tipo e tamanho).
* `list_allowed_directories()`: Raízes permitidas pelo harness.
* `create_directory(path: string)`: Criação de diretórios.
* `write_file(path: string, content: string)`: Criação ou atualização de arquivos.
* `edit_file(path: string, old_string: string, new_string: string)`: Substituição da primeira ocorrência.
* `move_file(source: string, destination: string)`: Mover/renomear arquivos.
* Aliases antigos seguem canônicos: `read_file`, `list_dir`, `search_code`, `execute_command`.

### Suíte 3: Memória e Estado (`memory_suite`)
* `core_memory_append(key: string, value: string)`: Adição na memória curta.
* `core_memory_replace(key: string, value: string)`: Atualização de chave existente.
* `archival_memory_insert(content: string)`: Inserção no repositório de longo prazo.
* `archival_memory_search(query: string)`: Recuperação contextual por busca.

### Suíte 4: Matemática, APIs e Computação Técnica (`technical_suite`)
* `python_interpreter(code: string)`: Execução de cálculos e algoritmos numéricos em Python.
* `calculator(expression: string)`: Avaliação direta de fórmulas matemáticas.
* `http_request(url: string, method: string, headers?: object, body?: object)`: Comunicação com endpoints REST.
* `database_query(query: string)`: Consultas SQL parametrizadas em bases de dados.

---

## 🔒 3. Sanitização de Segurança e PII (Bloqueadores Estritos)

1. **Bloqueio de Comandos Ofensivos / Malware:**
   * Qualquer linha de comando contendo as seguintes palavras-chave ou intenções deve ser **sumariamente rejeitada**:
     `mimikatz`, `dumpwcm`, `kerberoast`, `bypass`, `hidden`, `executionpolicy bypass`, `invoke-expression`, `iex(`, `downloadstring`, `c2`, `payload`, `meterpreter`, `defender bypass`, `amsi bypass`, `reverse shell`.
   * **Foco Permitido:** Diagnósticos de rede (`Test-NetConnection`, `Resolve-DnsName`), monitoramento de serviços (`Get-Service`, `Restart-Service`), auditoria de logs de eventos e automação DevOps.

2. **Eliminação de Informações Pessoais (PII):**
   * Nenhum dado real de pessoas físicas, escolas ou menores de idade deve constar nos dados.
   * Amostras de memória devem ser transformadas em cenários enterprise abstratos (ex: engenheiro de software gerenciando backlog de projeto, analista financeiro acompanhando portfólio corporativo).

---

## ⚖️ 4. Balanceamento de Tokens e Amostragem

* **Teto Absoluto:** $\le 5.000$ tokens (medido no tokenizer oficial `checkpoints/qwen3loop_tool_sft_final`).
* **Distribuição Saudável de Comprimento:**
  * Curto (100 a 800 tokens): 30% das amostras (chamadas atômicas e diretas).
  * Médio (800 a 2.000 tokens): 50% das amostras (inspeção + ação + relatório).
  * Longo (2.000 a 4.500 tokens): 20% das amostras (trajetórias de múltiplos turnos com retrajetória de erro).
* **Meta de Retrajetórias:** Mínimo de 10% a 15% do total de amostras contendo erros reais de ambiente (ex: arquivo não encontrado, parâmetro de API inválido, comando Linux no Windows) seguidos de reflexão corretiva e sucesso.
