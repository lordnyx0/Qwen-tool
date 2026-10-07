"""
tools_schema.py
Catalogo Canonico (27 ferramentas) — 02 Diretrizes §2.
MCP canonico desde 2026-10-07 (decisao do dono): os nomes MCP
(read_text_file, list_directory, search_files, ...) sao canonicos;
os nomes antigos (read_file, list_dir, search_code) seguem como
aliases canonicos (ambos aparecem no PURIFIED). Sem hard distractors.
"""

from typing import Dict, Any

TOOL_CATALOG: Dict[str, Dict[str, Any]] = {
    # ===== Suíte 1: Terminal & Shell (2) =====
    "bash": {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Executa comandos bash em terminal Linux e retorna stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Linha de comando bash"}
                },
                "required": ["command"],
            },
        },
    },
    "powershell": {
        "type": "function",
        "function": {
            "name": "powershell",
            "description": "Administracao legitima Windows (Get-Process, Get-Service, Get-ChildItem, Select-String, Test-NetConnection). Sem red team.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Comando PowerShell"}
                },
                "required": ["command"],
            },
        },
    },
    # ===== Suíte 2: Workspace & Codigo (5) =====
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Reads the content of a single file. Never use for directories - directories always go to list_dir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (never a directory)"}
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Cria ou atualiza arquivo com conteudo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho de destino"},
                    "content": {"type": "string", "description": "Conteudo do arquivo"},
                },
                "required": ["path", "content"],
            },
        },
    },
    "list_dir": {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "Lists the structured contents of a directory. Always use when the target is a directory; never use read_file for directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Caminho do diretorio"},
                    "recursive": {"type": "boolean", "description": "Listagem recursiva"},
                },
                "required": ["path"],
            },
        },
    },
    "search_code": {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Busca literal ou semantica no codigo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termo de busca"},
                    "path": {"type": "string", "description": "Escopo da busca"},
                },
                "required": ["query"],
            },
        },
    },
    "execute_command": {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": "Executa comando em sandbox local.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Comando sandbox"}
                },
                "required": ["command"],
            },
        },
    },
    # ===== Suite 2b: Workspace MCP (12, canonico desde 2026-10-07) =====
    "read_text_file": {
        "type": "function",
        "function": {
            "name": "read_text_file",
            "description": "Reads a text file, with optional head/tail/offset slicing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "head": {"type": "integer", "description": "First N lines"},
                    "tail": {"type": "integer", "description": "Last N lines"},
                    "offset": {"type": "integer", "description": "Start line"},
                },
                "required": ["path"],
            },
        },
    },
    "read_media_file": {
        "type": "function",
        "function": {
            "name": "read_media_file",
            "description": "Sniffs a media/binary file and returns its type and size.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    },
    "read_multiple_files": {
        "type": "function",
        "function": {
            "name": "read_multiple_files",
            "description": "Reads several files, concatenated with path headers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {"type": "array", "description": "File paths", "items": {"type": "string"}},
                },
                "required": ["paths"],
            },
        },
    },
    "list_directory": {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "Lists a directory, entries sorted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": ["path"],
            },
        },
    },
    "list_directory_with_sizes": {
        "type": "function",
        "function": {
            "name": "list_directory_with_sizes",
            "description": "Lists a directory with byte sizes, sorted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": ["path"],
            },
        },
    },
    "directory_tree": {
        "type": "function",
        "function": {
            "name": "directory_tree",
            "description": "Indented sorted tree of a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": ["path"],
            },
        },
    },
    "search_files": {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Grep-like literal search, file:line:match output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Search pattern"},
                    "path": {"type": "string", "description": "Search scope"},
                },
                "required": ["pattern"],
            },
        },
    },
    "get_file_info": {
        "type": "function",
        "function": {
            "name": "get_file_info",
            "description": "File or directory metadata (type and size).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path"},
                },
                "required": ["path"],
            },
        },
    },
    "list_allowed_directories": {
        "type": "function",
        "function": {
            "name": "list_allowed_directories",
            "description": "Lists harness-allowed roots. Takes no arguments.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    "create_directory": {
        "type": "function",
        "function": {
            "name": "create_directory",
            "description": "Creates a directory (mkdir -p).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                },
                "required": ["path"],
            },
        },
    },
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replaces first occurrence of old_string with new_string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_string": {"type": "string", "description": "Text to replace"},
                    "new_string": {"type": "string", "description": "Replacement"},
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["old_string", "new_string", "path"],
            },
        },
    },
    "move_file": {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Moves or renames a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source path"},
                    "destination": {"type": "string", "description": "Destination path"},
                },
                "required": ["source", "destination"],
            },
        },
    },
    # ===== Suíte 3: Memoria e Estado (4) =====
    "core_memory_append": {
        "type": "function",
        "function": {
            "name": "core_memory_append",
            "description": "Adiciona chave na memoria curta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Chave de memoria"},
                    "value": {"type": "string", "description": "Valor"},
                },
                "required": ["key", "value"],
            },
        },
    },
    "core_memory_replace": {
        "type": "function",
        "function": {
            "name": "core_memory_replace",
            "description": "Atualiza chave existente na memoria curta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Chave existente"},
                    "value": {"type": "string", "description": "Novo valor"},
                },
                "required": ["key", "value"],
            },
        },
    },
    "archival_memory_insert": {
        "type": "function",
        "function": {
            "name": "archival_memory_insert",
            "description": "Insere conteudo no repositorio de longo prazo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Conteudo"}
                },
                "required": ["content"],
            },
        },
    },
    "archival_memory_search": {
        "type": "function",
        "function": {
            "name": "archival_memory_search",
            "description": "Recupera contexto por busca. Args vazio {} = varredura completa (padrao legado retrieve_all).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Consulta de busca (omitivel para dump completo)"}
                },
                "required": [],
            },
        },
    },
    # ===== Suíte 4: Matematica/APIs (4) =====
    "python_interpreter": {
        "type": "function",
        "function": {
            "name": "python_interpreter",
            "description": "Executa codigo Python em sandbox e retorna stdout. Use print() no resultado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Codigo Python valido"}
                },
                "required": ["code"],
            },
        },
    },
    "calculator": {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Avalia formula matematica direta.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Expressao matematica"}
                },
                "required": ["expression"],
            },
        },
    },
    "http_request": {
        "type": "function",
        "function": {
            "name": "http_request",
            "description": "Chama endpoint REST.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL completa"},
                    "method": {"type": "string", "description": "GET ou POST"},
                },
                "required": ["url"],
            },
        },
    },
    "database_query": {
        "type": "function",
        "function": {
            "name": "database_query",
            "description": "SELECT somente-leitura em banco relacional.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT valido"}
                },
                "required": ["query"],
            },
        },
    },
}

DOMAINS = {
    "terminal": ["bash", "powershell"],
    "workspace": ["read_file", "write_file", "list_dir", "search_code", "execute_command",
                  "read_text_file", "read_media_file", "read_multiple_files",
                  "list_directory", "list_directory_with_sizes", "directory_tree",
                  "search_files", "get_file_info", "list_allowed_directories",
                  "create_directory", "edit_file", "move_file"],
    "memory": ["core_memory_append", "core_memory_replace", "archival_memory_insert", "archival_memory_search"],
    "technical": ["python_interpreter", "calculator", "http_request", "database_query"],
}

# Mapa legado -> canonico para migracao manual do PURIFIED (Fase C/D).
# technical_api diversa (toolmind/hermes, ~1000 tools) mantida como excecao documentada da matriz 06.
# Nomes antigos read_file/list_dir/search_code seguem canonicos como aliases dos MCP
# (read_text_file/list_directory/search_files); LEGACY_MAP abaixo e so para fontes terceiras.
LEGACY_MAP = {
    "filesystem-context": "read_file",
    "safe-exec": "execute_command",
    "terminal": "execute_command",
    "browser": "http_request",
    "browser_open": "http_request",
    "web_search": "http_request",
    "memory_get": "archival_memory_search",
    "memory_append": "core_memory_append",
    "memory_replace": "core_memory_replace",
    "memory_clear": "core_memory_replace",
    "memory_retrieve": "archival_memory_search",
    "memory_update": "core_memory_replace",
    "execute_python": "python_interpreter",
    "query_database_sql": "database_query",
}
