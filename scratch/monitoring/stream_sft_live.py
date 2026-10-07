import subprocess
import os
import sys
import time
import threading

env = os.environ.copy()
t = subprocess.run(["kaggle", "auth", "print-access-token"], capture_output=True, text=True, encoding="utf-8")
if t.returncode == 0:
    env["KAGGLE_API_TOKEN"] = t.stdout.strip()
env["PYTHONIOENCODING"] = "utf-8"

proc = subprocess.Popen(
    ["kaggle", "kernels", "logs", "-f", "lordnyx0/qwen3loop-tool-sft"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=env,
    text=True,
    encoding="utf-8",
    errors="replace"
)

lines = []

def reader():
    try:
        for line in iter(proc.stdout.readline, ""):
            lines.append(line)
    except Exception:
        pass

th = threading.Thread(target=reader, daemon=True)
th.start()

# Aguardar 25s para drenar o buffer completo do servidor
time.sleep(25)

try:
    proc.terminate()
    proc.wait(timeout=2)
except Exception:
    try:
        proc.kill()
    except Exception:
        pass

output_text = "".join(lines)
with open("logs/sft_live_stream.log", "w", encoding="utf-8") as f:
    f.write(output_text)

print(f"Total lines captured: {len(lines)}")
step_lines = [l.strip() for l in lines if "Passo [" in l or "Validação" in l]
print("Passos encontrados:")
for sl in step_lines:
    print(" ", sl)
