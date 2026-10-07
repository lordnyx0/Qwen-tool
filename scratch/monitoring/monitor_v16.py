import subprocess
import os
import time
import json
import sys

# Forçar stdout e stderr em UTF-8 com fallback para evitar erros de charmap cp1252 no Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

token = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = token
env['PYTHONIOENCODING'] = 'utf-8'

print("=== MONITOR DE EXECUÇÃO AO VIVO - KERNEL V16 (V15 CORRIGIDA) ===", flush=True)

log_file = 'logs/kaggle_v16_live.log'
last_len = 0

while True:
    time.sleep(5)
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
            
    # Verificar status do kernel
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
