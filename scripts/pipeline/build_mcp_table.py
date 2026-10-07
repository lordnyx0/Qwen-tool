"""build_mcp_table.py — tabela MCP x catalogo com contagens vivas.

MCP canonico desde 2026-10-07: `ours` e o proprio nome MCP.
Uso: python scripts/pipeline/build_mcp_table.py
Leitura do PURIFIED; escreve data/tool_catalog/mcp_tools.{json,csv}.
"""
import csv
import json
import os
from collections import Counter

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MCP = [
    # (mcp, readOnly, ours, output_contract, collector)
    ("read_text_file", True, "read_text_file",
     "verbatim content (optional head/tail slicing)", "collect: cat/head/tail"),
    ("read_media_file", True, "read_media_file",
     "TYPE N bytes: path (TYPE=PNG/SVG/BIN by magic)", "collect: magic sniff"),
    ("read_multiple_files", True, "read_multiple_files",
     "concat of '--- path ---\\n' + content per file", "collect: sequential cat"),
    ("list_directory", True, "list_directory",
     "[FILE] x / [DIR] y per entry, sorted", "collect: readdir+filetype"),
    ("list_directory_with_sizes", True, "list_directory_with_sizes",
     "name: N bytes per entry, sorted", "collect: stat"),
    ("directory_tree", True, "directory_tree",
     "indented tree, sorted", "collect: walk"),
    ("search_files", True, "search_files",
     "file:line:match (grep -rn)", "collect: grep -rn"),
    ("get_file_info", True, "get_file_info",
     "path: file|dir, N bytes (no mtime)", "collect: stat"),
    ("list_allowed_directories", True, "list_allowed_directories",
     "allowed roots (harness-defined)", "declared only"),
    ("create_directory", False, "create_directory",
     "created {path}", "collect: mkdir -p"),
    ("write_file", False, "write_file",
     "wrote N bytes to {path}", "collect: write"),
    ("edit_file", False, "edit_file",
     "replaced 1 occurrence in {path}", "collect: replace-first"),
    ("move_file", False, "move_file",
     "moved {src} -> {dst}", "collect: mv"),
]


def main():
    calls, decls = Counter(), Counter()
    with open(os.path.join(ROOT, "data", "cleaned",
                           "curriculum_3000_tool_sft_PURIFIED.jsonl"),
              encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            e = json.loads(line)
            for t in e.get("tools", []):
                decls[t["function"]["name"]] += 1
            for m in e.get("messages", []):
                for tc in (m.get("tool_calls") or []):
                    calls[tc["function"]["name"]] += 1

    rows = []
    for mcp, ro, ours, contract, collector in MCP:
        c = calls.get(mcp, 0)
        if c > 0:
            status = "covered"
        elif ours and calls.get(ours, 0):
            status = "legacy-name"
        elif ours and decls.get(ours, 0):
            status = "nominal"
        else:
            status = "gap"
        rows.append({"mcp_tool": mcp, "readOnlyHint": ro, "ours": ours,
                     "output_contract": contract, "collector": collector,
                     "calls_mcp_name": c,
                     "calls_ours_name": calls.get(ours, 0) if ours else 0,
                     "decls_ours_name": decls.get(ours, 0) if ours else 0,
                     "status": status})

    outdir = os.path.join(ROOT, "data", "tool_catalog")
    with open(os.path.join(outdir, "mcp_tools.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    with open(os.path.join(outdir, "mcp_tools.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['mcp_tool']}: ours={r['ours']} calls_mcp={r['calls_mcp_name']} "
              f"calls_ours={r['calls_ours_name']} decls_ours={r['decls_ours_name']} [{r['status']}]")


if __name__ == "__main__":
    main()
