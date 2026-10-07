"""Scratch: setup do sandbox do gate (recriavel)."""
import os
import shutil

ROOT = "C:/Users/Nyx/AppData/Local/Temp/opencode/gate_sandbox"
shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(ROOT + "/docs", exist_ok=True)
os.makedirs(ROOT + "/data", exist_ok=True)
open(ROOT + "/serve_gate.bat", "w").write(
    "@echo off\nREM servidor local\nset PORT=8080\n"
    '"%LLAMA_SERVER%" --host 127.0.0.1 --port %PORT%\n')
open(ROOT + "/docs/note.txt", "w").write("hello gate\nsecond line\n")
open(ROOT + "/data/a.txt", "w").write("alpha\n")
open(ROOT + "/data/b.txt", "w").write("beta\n")
print("OK sandbox:", ROOT)
