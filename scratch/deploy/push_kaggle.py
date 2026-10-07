import subprocess
import os
import sys

# 1. Obter o token de autenticacao OAuth ativo do Kaggle CLI
token = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = token
env['PYTHONIOENCODING'] = 'utf-8'

print(f"Token Kaggle autenticado com sucesso (prefixo {token[:8]}...)")

# 2. Upload do dataset atualizado lordnyx0/qwen3loop-tool-sft-data com -r tar
print("\n--- FAZENDO UPLOAD DA NOVA VERSÃO DO DATASET ---")
cmd_ds = [
    'kaggle', 'datasets', 'version',
    '-p', 'kaggle_sft_data',
    '-m', 'v15 clean: 2366 samples <=6144 tokens, implicit causal attention, chunked cross-entropy',
    '-r', 'tar'
]
res_ds = subprocess.run(cmd_ds, env=env, capture_output=True, text=True, encoding='utf-8')
print("STDOUT:", res_ds.stdout)
print("STDERR:", res_ds.stderr)
if res_ds.returncode != 0:
    print(f"Falha no upload do dataset (exit code {res_ds.returncode})")
    sys.exit(1)

# 3. Push do notebook kernel atualizado lordnyx0/qwen3loop-tool-sft
print("\n--- ENVIANDO KERNEL NOTEBOOK ATUALIZADO PARA O KAGGLE ---")
cmd_kernel = [
    'kaggle', 'kernels', 'push',
    '-p', 'kaggle_kernel'
]
res_k = subprocess.run(cmd_kernel, env=env, capture_output=True, text=True, encoding='utf-8')
print("STDOUT:", res_k.stdout)
print("STDERR:", res_k.stderr)
if res_k.returncode != 0:
    print(f"Falha no envio do kernel (exit code {res_k.returncode})")
    sys.exit(1)

print("\nDeploy da versão v15 concluído com sucesso!")
