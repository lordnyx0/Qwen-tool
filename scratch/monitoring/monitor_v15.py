import subprocess
import os
import time
import json
import sys

t = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = t
env['PYTHONIOENCODING'] = 'utf-8'

print("=== MONITOR DE EXECUÇÃO AO VIVO - KERNEL V15 (NATIVE CHECKPOINTING & TURING SDPA) ===")

# 1. Esperar transição de QUEUED para RUNNING
while True:
    st_res = subprocess.run(
        ['kaggle', 'kernels', 'status', 'lordnyx0/qwen3loop-tool-sft'],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    st = st_res.stdout.strip()
    print(f"Status atual: {st}", flush=True)
    if 'RUNNING' in st:
        print("Kernel v15 iniciou execução na infraestrutura Kaggle Dual Tesla T4!\n", flush=True)
        break
    if 'COMPLETE' in st or 'ERROR' in st or 'CANCEL' in st:
        print(f"Kernel finalizou: {st}", flush=True)
        break
    time.sleep(10)

# 2. Monitorar logs continuamente até vermos os passos de treino com loss real e VRAM
seen_chars = 0
step_count = 0
for _ in range(120):  # até 20-25 minutos de monitoramento contínuo
    res = subprocess.run(
        ['kaggle', 'kernels', 'logs', 'lordnyx0/qwen3loop-tool-sft'],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    raw = res.stdout
    if len(raw) > seen_chars:
        new_text = raw[seen_chars:]
        seen_chars = len(raw)
        with open('logs/kaggle_v15_live.log', 'w', encoding='utf-8') as f:
            f.write(raw)

        for line in new_text.splitlines():
            cleaned = line.lstrip(',').strip()
            if cleaned.startswith('{'):
                try:
                    d = json.loads(cleaned).get('data', '')
                    d_s = d.strip()
                    if d_s:
                        if any(k in d_s for k in ['Passo', 'GPU', 'Otimizador', 'Total de', 'Loss', 'Epoch', 'ETA', 'VRAM', 'Assertion', 'Traceback', 'Error', 'CUDA out of memory', 'RuntimeError', 'Backends SDPA']):
                            print(d_s, flush=True)
                            if 'Passo [' in d_s:
                                step_count += 1
                except Exception:
                    pass
            elif any(k in cleaned for k in ['Passo', 'GPU', 'Otimizador', 'Total de', 'Loss', 'Epoch', 'ETA', 'VRAM', 'Assertion', 'Traceback', 'Error', 'RuntimeError', 'Backends SDPA']):
                print(cleaned, flush=True)

    if step_count >= 5:
        print(f"\n[SUCESSO COMPROVADO NO KAGGLE] {step_count} passos de treino validados com sucesso no Kaggle Dual T4!", flush=True)
        break

    time.sleep(10)
