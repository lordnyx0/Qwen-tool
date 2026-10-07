import subprocess
import os
import sys

token = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = token
env['PYTHONIOENCODING'] = 'utf-8'

print(f"Token Kaggle autenticado com sucesso (prefixo {token[:8]}...)")

# Push do notebook kernel atualizado lordnyx0/qwen3loop-tool-sft
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

print("\nDeploy da versao v15 no Kaggle concluido com sucesso!")
