import json

SYS = "You are a helpful AI assistant equipped with specialized tools. Use them when needed."
# descriptions: PT legado (read_file/list_dir afiados em EN na V3.1); o harness só avalia name/args
T = {
 "bash": ("Executa comandos bash em terminal Linux e retorna stdout/stderr.",
           {"command": "Linha de comando bash"}),
 "powershell": ("Administracao legitima Windows. Sem red team.",
                {"command": "Comando PowerShell"}),
  "read_file": ("Reads the content of a single file. Never use for directories - directories always go to list_dir.",
                {"path": "File path (never a directory)"}),
  "list_dir": ("Lists the structured contents of a directory. Always use when the target is a directory; never use read_file for directories.",
               {"path": "Directory path"}),
 "core_memory_append": ("Adiciona chave na memoria curta.",
                        {"key": "Chave de memoria", "value": "Valor"}),
 "archival_memory_search": ("Recupera contexto por busca.",
                            {"query": "Consulta de busca"}),
 "python_interpreter": ("Executa codigo Python em sandbox e retorna stdout.",
                        {"code": "Codigo Python"}),
 "calculator": ("Avalia formula matematica direta.",
                {"expression": "Expressao matematica"}),
 "http_request": ("Chama endpoint REST.",
                  {"url": "URL completa", "method": "GET ou POST"}),
 "database_query": ("SELECT somente-leitura.",
                    {"query": "SQL SELECT"}),
}

def tool(n):
    d, props = T[n]
    return {"type": "function", "function": {"name": n, "description": d,
            "parameters": {"type": "object",
                           "properties": {k: {"type": "string", "description": v} for k, v in props.items()},
                           "required": list(props)}}}

def item(i, level, cat, tools, user, etool=None, eargs=None, out="", text=""):
    return {"id": "probe_canon_%03d" % i, "level": level, "category": cat,
            "tools": [tool(t) for t in tools], "tool_names": tools,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
            "expected_tool": etool, "expected_args": eargs or {},
            "ground_truth_stdout": out, "expected_text": text}

L1, L2, L3 = "level_1_foundation", "level_2_discrimination", "level_3_reasoning_python"
items = [
 item(1, L1, "single_tool_precise", ["bash"], "Show the top 5 CPU-consuming processes on this Linux host.",
      "bash", {"command": "ps aux --sort=-%cpu | head -n 5"}),
 item(2, L1, "single_tool_precise", ["bash"], "How much disk space is free on the root filesystem?",
      "bash", {"command": "df -h /"}),
 item(3, L1, "single_tool_precise", ["powershell"], "Check the status of the Background Intelligent Transfer service.",
      "powershell", {"command": "Get-Service -Name 'bits'"}),
 item(4, L1, "single_tool_precise", ["powershell"], "List all files in C:\\Logs including hidden ones.",
      "powershell", {"command": "Get-ChildItem -Path 'C:\\Logs' -Force"}),
 item(5, L1, "single_tool_precise", ["read_file"], "Read the application config at config/app.yaml.",
      "read_file", {"path": "config/app.yaml"}),
 item(6, L1, "single_tool_precise", ["read_file", "list_dir"], "List the contents of the docs/ directory.",
      "list_dir", {"path": "docs/"}),
 item(7, L1, "single_tool_precise", ["archival_memory_search"], "Do I have any notes about invoice approval thresholds?",
      "archival_memory_search", {"query": "invoice approval threshold"}),
 item(8, L1, "single_tool_precise", ["core_memory_append"], "Remember that the project deadline is 2024-12-01.",
      "core_memory_append", {"key": "project_deadline", "value": "2024-12-01"}),
 item(9, L1, "single_tool_precise", ["http_request"], "Check whether https://api.example.com/healthz is responding.",
      "http_request", {"url": "https://api.example.com/healthz", "method": "GET"}),
 item(10, L1, "single_tool_precise", ["calculator"], "What is (250 times 12) plus 99?",
      "calculator", {"expression": "(250*12)+99"}),
 item(11, L2, "multi_tool_discrimination", ["bash", "read_file"], "Show disk usage and read the file /etc/hosts.",
      "bash", {"command": "df -h"}),
 item(12, L2, "multi_tool_discrimination", ["powershell", "bash"], "On this Windows host, list running processes sorted by CPU.",
      "powershell", {"command": "Get-Process | Sort-Object CPU -Descending"}),
 item(13, L2, "multi_tool_discrimination", ["core_memory_append", "archival_memory_search"],
      "Search my notes for anything about the Lisbon trip.",
      "archival_memory_search", {"query": "Lisbon trip"}),
 item(14, L2, "multi_tool_discrimination", ["http_request", "database_query"],
      "How many active users are in the analytics database?",
      "database_query", {"query": "SELECT count(*) FROM users WHERE active = 1"}),
 item(15, L2, "multi_tool_discrimination", ["python_interpreter", "calculator"],
      "Compute 17 squared plus 5 using code.",
      "python_interpreter", {"code": "print(17**2 + 5)"}, out="294"),
 item(16, L3, "python_algorithmic_reasoning", ["python_interpreter"],
      "Compute 7 to the power 100 modulo 13 using Python.",
      "python_interpreter", {"code": "print(pow(7, 100, 13))"}, out="9"),
 item(17, L3, "python_algorithmic_reasoning", ["python_interpreter"],
      "How many digits are in 20 factorial? Use Python.",
      "python_interpreter", {"code": "import math\nprint(len(str(math.factorial(20))))"}, out="19"),
 item(18, L3, "python_algorithmic_reasoning", ["python_interpreter"],
      "Compute the greatest common divisor of 1071 and 462 with Python.",
      "python_interpreter", {"code": "import math\nprint(math.gcd(1071, 462))"}, out="21"),
 item(19, L3, "python_algorithmic_reasoning", ["python_interpreter"],
      "What is the 25th Fibonacci number (starting 0, 1)? Use Python.",
      "python_interpreter", {"code": "a, b = 0, 1\nfor _ in range(25):\n    a, b = b, a + b\nprint(a)"}, out="75025"),
 item(20, L3, "python_algorithmic_reasoning", ["python_interpreter"],
      "Sum all prime numbers below 50 with Python.",
      "python_interpreter", {"code": "print(sum(n for n in range(50) if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))))"}, out="328"),
 item(21, L1, "negative_abstention", ["bash"], "What is the capital of Canada?",
      None, {}, text="Ottawa"),
 item(22, L1, "negative_abstention", ["bash"], "Who wrote the novel Dom Casmurro?",
      None, {}, text="Machado de Assis"),
 item(23, L1, "negative_abstention", ["powershell"], "Explain photosynthesis in one sentence.",
      None, {}, text="photosynthesis"),
 item(24, L1, "negative_abstention", ["read_file"], "Write a haiku about rain.",
      None, {}, text="rain"),
 item(25, L1, "negative_abstention", ["archival_memory_search"], "What does the idiom break a leg mean?",
      None, {}, text="good luck"),
]

with open("data/eval_probe_canonical_25.jsonl", "w", encoding="utf-8") as f:
    for r in items:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

# autovalidação: parse + stdout L3 executado de verdade
import subprocess
rows = [json.loads(l) for l in open("data/eval_probe_canonical_25.jsonl", encoding="utf-8")]
assert len(rows) == 25 and len(set(r["id"] for r in rows)) == 25
for r in rows:
    assert [m["role"] for m in r["messages"]] == ["system", "user"], r["id"]
    if r["ground_truth_stdout"]:
        got = subprocess.run(["python", "-c", r["expected_args"]["code"]],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        assert got == r["ground_truth_stdout"], (r["id"], got)
        print(r["id"], "stdout OK:", got)
print("probe OK: 25 itens, uuids únicos, system/user, L3 verificado por execução")
