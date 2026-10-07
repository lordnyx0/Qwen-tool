import subprocess
import os
import time
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

token = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = token
env['PYTHONIOENCODING'] = 'utf-8'

print("=== MONITOR DE EXECUÇÃO AO VIVO - KERNEL V17 (TRUE SDPA DISPATCH) ===", flush=True)

# 1. Esperar transição de QUEUED para RUNNING
while True:
    st_res = subprocess.run(
        ['kaggle', 'kernels', 'status', 'lordnyx0/qwen3loop-tool-sft'],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    st = st_res.stdout.strip()
    print(f"Status atual: {st}", flush=True)
    if 'RUNNING' in st:
        print("Kernel v17 iniciou execução no Kaggle Dual Tesla T4!\n", flush=True)
        break
    if 'COMPLETE' in st or 'ERROR' in st or 'CANCEL' in st:
        print(f"Kernel finalizou: {st}", flush=True)
        break
    time.sleep(10)

log_file = 'logs/kaggle_v17_live.log'
last_len = 0

while True:
    time.sleep(6)
    res = subprocess.run(
        ['kaggle', 'kernels', 'logs', 'lordnyx0/qwen3loop-tool-sft'],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    lines = res.stdout.splitlines()
    if len(lines) > last_len:
        new_lines = lines[last_len:]
        for line in new_lines:
            if line.startswith('{"stream_name"') or line.startswith(',{"stream_name"'):
                try:
                    clean = line.lstrip(',')
                    d = json.loads(clean)
                    data = d.get('data', '')
                    if data.strip():
                        print(data.rstrip('\r\n'), flush=True)
                except Exception:
                    print(line, flush=True)
            else:
                print(line, flush=True)
        last_len = len(lines)
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(res.stdout)
            
    st_res = subprocess.run(
        ['kaggle', 'kernels', 'status', 'lordnyx0/qwen3loop-tool-sft'],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    cur_st = st_res.stdout.strip()
    if 'COMPLETE' in cur_st or 'ERROR' in cur_st or 'CANCEL' in cur_st:
        print(f"\nKernel finalizou com status: {cur_st}", flush=True)
        break
