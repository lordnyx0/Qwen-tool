"""harness_local.py - execucao REAL de tools locais em jail (fixture_fs).

Uso: python scripts/pipeline/harness_local.py --calls <calls.jsonl>
       --out <fixtures.jsonl> [--root data/staging/v37_templates/fixture_fs]
Cada linha de calls: {"id": str, "tool": read_text_file|list_directory|
search_files|create_directory|edit_file|move_file|read_file|list_dir|
powershell|execute_python, "arguments": {...}}. Nomes antigos read_file/list_dir
sao aliases (mesmos runners). Saida por linha: {"id","tool","arguments",
"ok","exit_code","stdout","stderr"} com textos VERBATIM (viram conteudo de
mensagens tool na autoria manual). list_dir retorna entradas ordenadas
(contrato do harness). Seguranca: jail de paths, blocklist, timeout 20s,
uso manual local (nao expor como servico).
"""
import argparse
import json
import os
import re
import subprocess
import sys

TIMEOUT = 20
PS_BLOCK = re.compile(r"(invoke-webrequest|invoke-restmethod|\birw?\b|\bcurl\b|"
                       r"invoke-expression|\biex\b|start-process|remove-item|del\s+"
                       r"|format-|stop-computer|restart-computer|new-service|"
                       r"set-executionpolicy)", re.I)
PY_BLOCK = re.compile(r"(os\.(remove|unlink|rename|renames|rmdir|mkdir|makedirs|"
                       r"system|popen|exec[lv])|shutil\.(rmtree|move|copy|copytree)|"
                       r"subprocess|socket|urllib|requests|importlib|__import__|"
                       r"open\s*\([^)]*['\"][wa+]|pathlib[^)]*\.(write_text|rename|"
                       r"unlink|mkdir|rmdir)|eval\s*\(|exec\s*\(|getattr\s*\()", re.I)


def _resolve(root, p):
    full = os.path.realpath(os.path.join(root, p or ""))
    base = os.path.realpath(root)
    if full != base and not full.startswith(base + os.sep):
        raise ValueError(f"fuga do jail: {p!r}")
    return full


def run_read_file(root, args):
    with open(_resolve(root, args["path"]), "r", encoding="utf-8") as f:
        return True, 0, f.read(), ""


def run_list_dir(root, args):
    entries = sorted(os.listdir(_resolve(root, args.get("path", "."))))
    return True, 0, "\n".join(entries) + ("\n" if entries else ""), ""


def run_read_text_file(root, args):
    with open(_resolve(root, args["path"]), "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    if args.get("head") is not None:
        lines = lines[:int(args["head"])]
    if args.get("tail") is not None:
        lines = lines[-int(args["tail"]):]
    if args.get("offset") is not None:
        lines = lines[int(args["offset"]):]
    text = "\n".join(lines)
    return True, 0, text + ("\n" if text else ""), ""


def run_read_media_file(root, args):
    full = _resolve(root, args["path"])
    with open(full, "rb") as f:
        head = f.read(8)
    if head[:4] == b"\x89PNG":
        kind = "PNG"
    elif head[:2] == b"BM" or b"<svg" in head.lower():
        kind = "SVG" if b"<svg" in head.lower() else "BIN"
    else:
        kind = "BIN"
    return True, 0, f"{kind} {os.path.getsize(full)} bytes: {args['path']}\n", ""


def run_read_multiple_files(root, args):
    parts = []
    for p in args["paths"]:
        with open(_resolve(root, p), "r", encoding="utf-8") as f:
            parts.append(f"--- {p} ---\n" + f.read())
    return True, 0, "\n".join(parts) + ("\n" if parts else ""), ""


def run_list_directory_with_sizes(root, args):
    full = _resolve(root, args.get("path", "."))
    rows = [f"{e}: {os.path.getsize(os.path.join(full, e))} bytes"
            for e in sorted(os.listdir(full))]
    return True, 0, "\n".join(rows) + ("\n" if rows else ""), ""


def run_directory_tree(root, args):
    out = []
    for base, dirs, files in os.walk(_resolve(root, args.get("path", "."))):
        dirs.sort()
        depth = os.path.relpath(base, root).count(os.sep)
        depth = 0 if os.path.relpath(base, root) == "." else depth + 1
        out.append("  " * depth + os.path.basename(base) + "/")
        for fn in sorted(files):
            out.append("  " * (depth + 1) + fn)
    return True, 0, "\n".join(out) + ("\n" if out else ""), ""


def run_search_files(root, args):
    pat = re.compile(args["pattern"])
    hits = []
    base = _resolve(root, args.get("path", "."))
    targets = [base] if os.path.isfile(base) else [
        os.path.join(b, fn) for b, _, fns in os.walk(base) for fn in sorted(fns)]
    for fp in targets:
        try:
            with open(fp, "r", encoding="utf-8", errors="strict") as f:
                for i, line in enumerate(f, 1):
                    if pat.search(line):
                        rel = os.path.relpath(fp, root).replace(os.sep, "/")
                        hits.append(f"{rel}:{i}:{line.rstrip()}")
        except (OSError, ValueError):
            continue
    return True, 0, "\n".join(hits) + ("\n" if hits else ""), ""


def run_get_file_info(root, args):
    full = _resolve(root, args["path"])
    kind = "dir" if os.path.isdir(full) else "file"
    return True, 0, f"{args['path']}: {kind}, {os.path.getsize(full)} bytes\n", ""


def run_list_allowed_directories(root, args):
    return True, 0, os.path.normpath(root) + "\n", ""


def run_create_directory(root, args):
    full = _resolve(root, args["path"])
    os.makedirs(full, exist_ok=True)
    return True, 0, f"created {args['path']}\n", ""


def run_edit_file(root, args):
    full = _resolve(root, args["path"])
    with open(full, "r", encoding="utf-8") as f:
        text = f.read()
    if args["old_string"] not in text:
        raise ValueError(f"old_string nao encontrada em {args['path']!r}")
    with open(full, "w", encoding="utf-8") as f:
        f.write(text.replace(args["old_string"], args["new_string"], 1))
    return True, 0, f"replaced 1 occurrence in {args['path']}\n", ""


def run_move_file(root, args):
    src = _resolve(root, args["source"])
    dst = _resolve(root, args["destination"])
    os.makedirs(os.path.dirname(dst) or root, exist_ok=True)
    os.replace(src, dst)
    return True, 0, f"moved {args['source']} -> {args['destination']}\n", ""


def run_powershell(root, args):
    cmd = args["command"]
    if PS_BLOCK.search(cmd):
        return False, 126, "", f"REFUSED by harness blocklist: {cmd[:80]}"
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive",
                        "-Command", cmd], cwd=root, stdin=subprocess.DEVNULL,
                       capture_output=True, timeout=TIMEOUT)
    out = r.stdout.decode("utf-8", errors="replace")
    err = r.stderr.decode("utf-8", errors="replace")
    return r.returncode == 0, r.returncode, out, err


def run_python(root, args):
    code = args["code"]
    if PY_BLOCK.search(code):
        return False, 126, "", "REFUSED by harness blocklist"
    r = subprocess.run([sys.executable, "-c", code], cwd=root,
                       stdin=subprocess.DEVNULL, capture_output=True, timeout=TIMEOUT)
    out = r.stdout.decode("utf-8", errors="replace")
    err = r.stderr.decode("utf-8", errors="replace")
    return r.returncode == 0, r.returncode, out, err


RUNNERS = {"read_file": run_read_file, "list_dir": run_list_dir,
           "read_text_file": run_read_text_file,
           "read_media_file": run_read_media_file,
           "read_multiple_files": run_read_multiple_files,
           "list_directory": run_list_dir,
           "list_directory_with_sizes": run_list_directory_with_sizes,
           "directory_tree": run_directory_tree,
           "search_files": run_search_files,
           "get_file_info": run_get_file_info,
           "list_allowed_directories": run_list_allowed_directories,
           "create_directory": run_create_directory,
           "edit_file": run_edit_file,
           "move_file": run_move_file,
           "powershell": run_powershell, "execute_python": run_python}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default="data/staging/v37_templates/fixture_fs")
    a = ap.parse_args()
    n_ok = n_err = 0
    with open(a.calls, encoding="utf-8") as fin, open(a.out, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            c = json.loads(line)
            fn = RUNNERS.get(c["tool"])
            try:
                if fn is None:
                    raise ValueError(f"tool nao suportada: {c['tool']}")
                ok, code, out, err = fn(a.root, c.get("arguments", {}))
            except Exception as e:  # erros reais (FileNotFound, decode, jail...)
                ok, code, out, err = False, 1, "", f"{type(e).__name__}: {e}"
            n_ok += ok
            n_err += (not ok)
            fout.write(json.dumps({"id": c["id"], "tool": c["tool"],
                                   "arguments": c.get("arguments", {}), "ok": ok,
                                   "exit_code": code, "stdout": out, "stderr": err},
                                  ensure_ascii=False) + "\n")
    print(f"OK={n_ok} ERR={n_err} -> {a.out}")


if __name__ == "__main__":
    main()
