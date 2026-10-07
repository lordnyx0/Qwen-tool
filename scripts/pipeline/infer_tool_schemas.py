"""infer_tool_schemas.py — C2 do plano v35 (ver 13_v35_plano_correcao.md).

Para defs com `properties` vazio que recebem calls COM args bem-sucedidas
(observação seguinte sem erro; dangling excluídas), infere `type` por valor
dentro da MESMA amostra (sem vazamento cross-sample):
  bool->boolean, int->integer, float->number, str->string, list->array,
  dict->object, None->ignorado; tipo misto -> mais frequente (empate: string).
`required` sempre omitido. Defs sem nenhum sucesso com args ficam intactas.
Uso: python scripts/pipeline/infer_tool_schemas.py [--apply]
Sem --apply: só relata. Com --apply: backup .bakN + reescreve PURIFIED.
"""
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PUR = os.path.join(ROOT, "data", "cleaned",
                   "curriculum_3000_tool_sft_PURIFIED.jsonl")
ERR = re.compile(r"ENOENT|EISDIR|Traceback|PermissionError|FileNotFoundError|"
                 r"NotJSONError|Invalid|failed|ERROR|Error:|does not exist|"
                 r"not found|No such|aborted")

JTYPES = {bool: "boolean", int: "integer", float: "number",
          str: "string", list: "array", dict: "object"}


def jtype(v):
    if isinstance(v, bool):
        return "boolean"
    for t, name in ((int, "integer"), (float, "number"), (str, "string"),
                    (list, "array"), (dict, "object")):
        if isinstance(v, t):
            return name
    return None


def main():
    apply = "--apply" in sys.argv
    rows = [json.loads(l) for l in open(PUR, encoding="utf-8") if l.strip()]

    def success_obs(msgs):
        """Rende (nome, args) de calls com sucesso (observação sem erro)."""
        for i, m in enumerate(msgs):
            for tc in (m.get("tool_calls") or []):
                try:
                    a = json.loads(tc["function"]["arguments"])
                except Exception:
                    continue
                if not isinstance(a, dict) or not a:
                    continue
                succ = False
                for m2 in msgs[i:]:
                    if m2.get("role") == "tool":
                        if not ERR.search(m2.get("content", "")):
                            succ = True
                        break
                if succ:
                    yield tc["function"]["name"], a

    glb = defaultdict(Counter)
    for r in rows:
        for name, a in success_obs(r["messages"]):
            for k, v in a.items():
                jt = jtype(v)
                if jt:
                    glb[(name, k)][jt] += 1

    n_empty = n_p1 = n_p2 = n_calls_used = 0
    for r in rows:
        msgs = r["messages"]
        for t in r["tools"]:
            fn = t["function"]
            props = (fn.get("parameters", {}) or {}).get("properties", None)
            if props:
                continue
            n_empty += 1
            obs = defaultdict(list)
            for name, a in success_obs(msgs):
                if name != fn["name"]:
                    continue
                n_calls_used += 1
                for k, v in a.items():
                    jt = jtype(v)
                    if jt:
                        obs[k].append(jt)
            src = "p1"
            if not obs:
                for (name, k), votes in glb.items():
                    if name == fn["name"]:
                        obs[k].extend([jt for jt, c in votes.items()
                                       for _ in range(c)])
                src = "p2"
            if not obs:
                continue
            new_props = {}
            for k, votes in obs.items():
                c = Counter(votes)
                top = c.most_common()
                best = top[0][0]
                if len(top) > 1 and top[0][1] == top[1][1]:
                    best = "string"
                new_props[k] = {"type": best}
            fn["parameters"] = {"type": "object", "properties": new_props}
            if src == "p1":
                n_p1 += 1
            else:
                n_p2 += 1
    print(f"defs vazias: {n_empty} | preenchidas p1: {n_p1} p2: {n_p2} | "
          f"calls-sucesso usadas: {n_calls_used}")
    if not apply:
        print("(dry-run; rode com --apply para gravar com backup)")
        return
    n = 1
    while os.path.exists(f"{PUR}.bak{n}"):
        n += 1
    shutil.copyfile(PUR, f"{PUR}.bak{n}")
    print("backup:", f"{PUR}.bak{n}")
    with open(PUR, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("PURIFIED reescrito.")


if __name__ == "__main__":
    main()
