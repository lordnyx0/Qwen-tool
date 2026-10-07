import json
import re

with open('logs/kaggle_v11_run.log', 'r', encoding='utf-8') as f:
    text = f.read()

# Parse JSON objects if it's JSON or regex find
for line in text.splitlines():
    if line.startswith('{') or line.startswith(',{'):
        cleaned = line.lstrip(',')
        try:
            obj = json.loads(cleaned)
            data = obj.get('data', '')
            if any(k in data for k in ['Passo', 'VRAM', 'Epoch', 'Loss', 'loss', 'Otimizador', 'Total de', 'GPU 0', 'GPU 1', 'precision', 'Allocated', 'CUDA out of memory']):
                print(f"[{obj.get('time', 0):.1f}s] {data.strip()}")
        except Exception:
            pass
