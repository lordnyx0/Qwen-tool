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

print("=== MONITOR DE EXECUÇÃO AO VIVO - KERNEL V18 (1 ÉPOCA, 5000 TOKENS) ===", flush=True)

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
        print("Kernel v18 iniciou execução no Kaggle Dual Tesla T4!\n", flush=True)
        break
    if 'COMPLETE' in st or 'ERROR' in st or 'CANCEL' in st:
        print(f"Kernel finalizou: {st}", flush=True)
        break
    time.sleep(8)

log_file = 'logs/kaggle_runs/kaggle_v18_live.log'
os.makedirs('logs/kaggle_runs', exist_ok=True)
last_len = 0

step_count = 0
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
    full_output = res.stdout
    if full_output:
        with open('logs/raw_v18.log', 'w', encoding='utf-8') as f:
            f.write(full_output)
            
        try:
            entries = json.loads(full_output)
            clean_lines = []
            for item in entries:
                d = item.get("data", "")
                if d:
                    clean_lines.append(d)
            text_log = "".join(clean_lines)
            
            if len(text_log) > last_len:
                new_chunk = text_log[last_len:]
                last_len = len(text_log)
                print(new_chunk, end="", flush=True)
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(new_chunk)
                    
            if "Passo [1/" in text_log and step_count == 0:
                step_count = 1
                print("\n[OK] Passo 1 completado com sucesso no Kernel v18!\n", flush=True)
        except Exception:
            pass

    # Checar se finalizou
    st_chk = subprocess.run(
        ['kaggle', 'kernels', 'status', 'lordnyx0/qwen3loop-tool-sft'],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    st_text = st_chk.stdout.strip()
    if 'COMPLETE' in st_text:
        print("\n\n>>> KERNEL V18 FINALIZADO COM SUCESSO! Checkpoint zip gerado! <<<", flush=True)
        break
    elif 'ERROR' in st_text or 'CANCEL' in st_text:
        print(f"\n\n>>> KERNEL ENCERROU: {st_text} <<<", flush=True)
        break
