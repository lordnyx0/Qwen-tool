"""Scratch: metricas de gate (duplicate-call, invented-path, unnecessary-tool)."""
import json
import re
from collections import Counter

STOP = set("what which that this with from have has are was were will would could should there their about into over under your you the and for are but not you all any can had her was one our out day get has him his how man new now old see two way who boy did its let put say she too use".split())

for temp in ("0.0", "0.8"):
    rep = json.load(open(f"C:/Users/Nyx/AppData/Local/Temp/opencode/gate_v38_t{temp}.json",
                         encoding="utf-8"))
    n_calls = n_dup = n_inv = n_unnec = 0
    det = []
    for case in rep:
        req = {"asfkap": "create a folder called asfkap", "enoent": "read arquivo_que_nao_existe.txt",
               "porta": "read serve_gate.bat", "pycalc": "print(13**7)", "hi": "hi!",
               "list": "list all files"}[case["id"]]
        rwords = set(w for w in re.findall(r"[a-z]{5,}", req.lower()) if w not in STOP)
        seen = set()
        for c in case["calls"]:
            call, res = c.split(" -> ", 1)
            n_calls += 1
            if call in seen:
                n_dup += 1
            seen.add(call)
            m = re.search(r"'([^']{2,60})'", call)
            if m:
                lit = m.group(1)
                if lit not in ("asfkap", "serve_gate.bat", ".", "./", "api", "notes", "media", "deep"):
                    if not any(w in lit.lower() for w in rwords):
                        n_inv += 1
                        det.append((case["id"], lit))
            if case["id"] == "hi":
                n_unnec += 1
    print(f"--- t={temp}: calls={n_calls} dup-rate={100*n_dup/max(n_calls,1):.1f}% "
          f"invented={n_inv} unnecessary(hi)={n_unnec}", flush=True)
    for d in det[:10]:
        print("   inv:", d, flush=True)
