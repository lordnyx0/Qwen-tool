# Qwen Control Center — Especificação

Programa único (.exe) que centraliza a dinâmica do projeto qwen:
SFT/DPO/GRPO, dataset, treino local e nuvem, evals e documentação.
Esta especificação é viva: muda junto com o código, nunca depois.

## 1. Objetivo

1.1. Uma pessoa que abre o programa sem contexto ativo entende em
menos de 1 minuto: o estado do dataset (N, gates), qual modelo é o
SHIP, o que está treinando onde, e o que fazer a seguir.
1.2. Toda ação executável pelo terminal (validar, treinar, avaliar,
versionar, push) é executável pelo programa, com os mesmos scripts
canônicos — o programa orquestra, nunca duplica lógica.
1.3. Toda verdade de estado (N do PURIFIED, gates, checkpoints, sessões)
é lida do repositório na hora, nunca de cache. Número exibido =
número medido.
1.4. Fora do programa: edição concorrente no PURIFIED, segredos no
código, treino sem backup `.bakN`, merge sem validação. O programa
impõe essas travas quando executa fluxos.

## 2. Arquitetura

2.1. Stack: Python + PySide6, empacotado com PyInstaller (one-dir;
one-file se o tamanho permitir). Sem servidor, sem banco: o
repositório `qwen/` é o banco de dados (arquivos + `Kaggle/`).
2.2. Camadas:
- `app/ui/` — janelas e abas (apresentação desacoplada e modular).
- `app/core/` — configuração declarativa (`config.py` com suporte a `workbench.json`),
  leitura de estado (`repo_state.py`: N, gates, modelos, VRAM, sessões Kaggle),
  caminhos portáveis (`paths.py`) e carregamento/taxonomia (`dataset_loader.py`).
- `app/workers/` — QThreads e processos desacoplados (`command_worker.py`),
  streaming/tail dinâmico de logs (`log_tail_worker.py`) e ciclo de vida
  multiplataforma de processos (`process_manager.py` via `subprocess.Popen` detached)
  sem travar a UI e sem dependência de WMI ou caminhos de usuário.
- Scripts canônicos (`scripts/`, `Kaggle/`) — intocados; o programa
  os invoca com os mesmos flags documentados no handover.
2.3. Abas (ordem de construção): 1 Estado · 2 Treinos · 3 Evals ·
4 Splits · 5 Docs.
2.4. Menu Superior: Arquivo -> Abrir Pasta do Projeto... (Ctrl+O) para alternar
entre múltiplos repositórios (ex: qwen, svg_generator) em tempo de execução;
Recarregar Projeto (F5).
2.5. Build: `build_exe.bat` e `QwenControlCenter.spec` com exclusão de módulos Qt
desnecessários para footprint enxuto.

## 3. Aba Estado
3.1. Lê na hora: PURIFIED N, `checkpoints/`, `models/gguf/`, VRAM
(nvidia-smi), servidores :8080–8086 (`/health`).
3.2. Kaggle só sob demanda (botão Atualizar chama `kaggle kernels
status` nos kernels conhecidos ou configurados, com `KAGGLE_API_TOKEN` em memória).
3.3. Proibido cache: todo Refresh re-executa as leituras.

## 4. Aba Treinos
4.1. Presets locais executam o script em subprocesso detached multiplataforma
com PID rastreado (`process_manager.py`), log dinâmico em `.cache/logs/` ou temp.
4.2. Suporte a presets configurados via `workbench.json` com fallback para os
padrões do repositório.
4.3. Tail de log contínuo assíncrono (`log_tail_worker.py`) com auto-scroll
e botão de interrupção forçada (`kill_process_tree`).

## 5. Aba Evals
5.1. Presets dinâmicos (lidos de `workbench.json` ou padrões locais):
Sonda Canônica 25, Bateria 121 (Q8 v38, CPU), Holdout 90 Baseline.
5.2. Histórico lista arquivos `.json` de resultados e exibe métricas agregadas.
5.3. Comparador de Runs: seleciona 2 avaliações e exibe delta de acurácia,
formato e lista regressões (passava no baseline e falhou no novo).
5.4. Playground Tool-Calling: teste rápido via router local (`:8080`) com
inspeção direta de `<tool_call>`.

## 6. Aba Splits
6.1. Filtros combináveis em duas linhas:
  - Linha 1: Executor, Padrão/Calls (`single`, `multi`, `retry`, `abstention`),
    Pilar/Categoria e Subpilar (alimentados dinamicamente via `get_taxonomy_categories()`).
  - Linha 2: Efeito Colateral (`read_only`, `mutating`, `none`), Max-Len, busca textual e ações.
6.2. Tabela Virtualizada: Implementada com `QTableView` e `SamplesTableModel(QAbstractTableModel)`,
renderizando milhares de amostras em < 0.1ms sem travamento de interface.
6.3. Visualizador in-app completo de amostras (split view):
  - Sub-aba 1: Diálogo formatado com tool calls e respostas.
  - Sub-aba 2: Schemas de ferramentas declaradas.
  - Sub-aba 3: JSON bruto integral.
  - Sub-aba 4: Renderizador Gráfico SVG (`QSvgWidget`), ativado automaticamente quando
    a amostra contém código vetorial `<svg>...</svg>`.
6.4. Materialização assíncrona de views via `CommandWorker` (`scripts/pipeline/make_view.py`)
com suporte ao flag `--effect`.

## 7. Aba Docs
7.1. Varredura recursiva universal de arquivos `.md` na pasta `docs/` e raiz de
qualquer repositório aberto, com botão de atualização dinâmica.

## 8. Convenções de manutenção
8.1. Spec-first: mudança de comportamento entra aqui antes ou junto do
código, nunca depois.
8.2. Gates do repo valem para o programa: `validate_docs.py` após mexer
em docs; `validate_sft.py` se tocar dado; sintaxe (`ast.parse`) antes
de qualquer push de kernel que consuma o script.
8.3. Single-writer: o programa nunca edita PURIFIED nem resultados
históricos; só materializa views e executa os scripts canônicos.
8.4. Segredos nunca em disco pelo programa (tokens só em memória).
8.5. Build: `build_exe.bat` (PyInstaller one-dir via spec). Testar o .exe com
repo real antes de entregar.

