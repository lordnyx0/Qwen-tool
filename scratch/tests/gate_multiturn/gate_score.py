"""Scratch: metricas para qualquer gate salvo."""
import json
import re
import sys

STOP = set("what which that this with from have has are was were will would could should there their about into over under your you the and for are but not you all any can had her was one our out day get has him his how man new now old see two way who boy did its let put say she too use".split())
REQ = {"asfkap": "create a folder called asfkap", "enoent": "read arquivo_que_nao_existe.txt",
       "porta": "read serve_gate.bat", "pycalc": "print(13**7)", "hi": "hi!",
       "list": "list all files"}

fp = sys.argv[1]
rep = json.load(open(fp, encoding="utf-8"))
n_calls = n_dup = n_inv = n_unnec = 0
for case in rep:
    rwords = set(w for w in re.findall(r"[a-z]{5,}", REQ[case["id"]].lower()) if w not in STOP)
    seen = set()
    for c in case["calls"]:
        call = c.split(" -> ", 1)[0]
        n_calls += 1
        if call in seen:
            n_dup += 1
        seen.add(call)
        m = re.search(r"'([^']{2,60})'", call)
        if m and not any(w in m.group(1).lower() for w in list(rwords) + ["asfkap", "serve_gate", "13**7"]):
            if m.group(1) not in (".", "./"):
                n_inv += 1
        if case["id"] == "hi":
            n_unnec += 1
print(f"{fp.split('/')[-1]}: calls={n_calls} dup={100*n_dup/max(n_calls,1):.1f}% inv={n_inv} hi-calls={n_unnec}")
for case in rep:
    print(f"  {case['id']}: {len(case['calls'])} calls | final: {case['final'][:90].replace(chr(10),' ')}")
