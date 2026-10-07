import subprocess
import os
import time
import json

t = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = t
env['PYTHONIOENCODING'] = 'utf-8'

print("=== INICIANDO MONITORAMENTO CONTÍNUO DO KERNEL V12 ===")

seen_lines = 0
for attempt in range(25):
    res = subprocess.run(
        ['kaggle', 'kernels', 'logs', 'lordnyx0/qwen3loop-tool-sft'],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    raw = res.stdout.strip()
    if raw:
        lines = raw.splitlines()
        if len(lines) > seen_lines:
            for l in lines[seen_lines:]:
                # Parse JSON if output is stream JSON or plain text
                cleaned = l.lstrip(',').strip()
                if cleaned.startswith('{'):
                    try:
                        d = json.loads(cleaned).get('data', '')
                        if d.strip():
                            print(d.strip())
                    except Exception:
                        print(cleaned)
                else:
                    print(l)
            seen_lines = len(lines)
            with open('logs/kaggle_v12_live.log', 'w', encoding='utf-8') as f:
                f.write(raw)

    # Check status
    st_res = subprocess.run(
        ['kaggle', 'kernels', 'status', 'lordnyx0/qwen3loop-tool-sft'],
        env=env,
        capture_output=True,
        text=True,
        encoding='utf-8'
    )
    st = st_res.stdout.strip()
    if 'COMPLETE' in st or 'ERROR' in st or 'CANCEL' in st:
        print(f"\nFinal do Kernel detectado: {st}")
        break

    time.sleep(15)
