import subprocess
import os
import sys

token = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = token
env['PYTHONIOENCODING'] = 'utf-8'

print(f"Token Kaggle autenticado com sucesso (prefixo {token[:8]}...)")

# 1. Upload da nova versão do dataset com -r tar
print("\n--- UPLOAD DO DATASET ATUALIZADO (TRUE SDPA + IMPLICIT CAUSAL MASK) ---")
cmd_ds = [
    'kaggle', 'datasets', 'version',
    '-p', 'kaggle_sft_data',
    '-m', 'v16: true sdpa dispatch, implicit causal mask, chunked ce',
    '-r', 'tar'
]
res_ds = subprocess.run(cmd_ds, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
print("STDOUT:", res_ds.stdout)
if res_ds.returncode != 0:
    print(f"Falha no upload do dataset (exit code {res_ds.returncode})")
    sys.exit(1)

# 2. Push do notebook kernel atualizado
print("\n--- ENVIANDO KERNEL NOTEBOOK ATUALIZADO PARA O KAGGLE ---")
cmd_kernel = [
    'kaggle', 'kernels', 'push',
    '-p', 'kaggle_kernel'
]
res_k = subprocess.run(cmd_kernel, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
print("STDOUT:", res_k.stdout)
if res_k.returncode != 0:
    print(f"Falha no envio do kernel (exit code {res_k.returncode})")
    sys.exit(1)

print("\nDeploy do kernel e dataset concluído com sucesso!")
