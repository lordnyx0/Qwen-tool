import json

with open('logs/kaggle_v12_full.log', 'r', encoding='utf-8') as f:
    text = f.read()

for line in text.splitlines():
    if line.startswith('{') or line.startswith(',{'):
        cleaned = line.lstrip(',')
        try:
            obj = json.loads(cleaned)
            data = obj.get('data', '')
            t = obj.get('time', 0)
            if any(k in data for k in ['Passo', 'VRAM', 'Epoch', 'Loss', 'loss', 'Otimizador', 'Total de', 'GPU 0', 'GPU 1', 'precision', 'Allocated', 'CUDA out of memory', 'Traceback', 'Error', 'accelerate launch', 'mixed_precision', 'dynamo_backend', 'ETA', 'checkpoint']):
                print(f"[{t:.1f}s] {data.strip()[:140]}")
        except Exception:
            pass
