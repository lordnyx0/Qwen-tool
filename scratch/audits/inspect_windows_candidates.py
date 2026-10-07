import sys
sys.stdout.reconfigure(encoding='utf-8')
from datasets import load_dataset

print("=== 1. Testando Kimberlyindiva/powershell-devops-mega-dataset ===")
try:
    ds = load_dataset('Kimberlyindiva/powershell-devops-mega-dataset', split='train', streaming=True)
    sample = next(iter(ds))
    print('Chaves:', list(sample.keys()))
    for k in list(sample.keys()):
        print(f"  {k}: {str(sample[k])[:150]}")
except Exception as e:
    print('Erro:', e)

print("\n=== 2. Testando xlangai/windows_osworld ===")
try:
    ds_win = load_dataset('xlangai/windows_osworld', split='train', streaming=True)
    sample_win = next(iter(ds_win))
    print('Chaves:', list(sample_win.keys()))
    for k in list(sample_win.keys())[:5]:
        print(f"  {k}: {str(sample_win[k])[:150]}")
except Exception as e:
    print('Erro:', e)
