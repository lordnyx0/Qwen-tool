import subprocess
import os
import sys
import zipfile
import time

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

dest_dir = "checkpoints/qwen3loop_tool_sft_v18"
os.makedirs(dest_dir, exist_ok=True)

print("=== BAIXANDO ARTEFATOS DO KERNEL V18 DO KAGGLE ===", flush=True)

token = subprocess.check_output(['kaggle', 'auth', 'print-access-token'], text=True).strip()
env = os.environ.copy()
env['KAGGLE_API_TOKEN'] = token
env['PYTHONIOENCODING'] = 'utf-8'

cmd = [
    'kaggle', 'kernels', 'output',
    'lordnyx0/qwen3loop-tool-sft',
    '-p', dest_dir
]
print(f"Executando download para {dest_dir}...", flush=True)
t0 = time.time()
res = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
print("STDOUT:", res.stdout, flush=True)
print("STDERR:", res.stderr, flush=True)

if res.returncode != 0:
    print(f"Erro no download (exit code {res.returncode})", flush=True)
    sys.exit(1)

print(f"Download concluído em {time.time()-t0:.1f}s!", flush=True)

# Listar arquivos baixados
files = os.listdir(dest_dir)
print("\nArquivos recebidos em", dest_dir, ":")
for f in files:
    f_path = os.path.join(dest_dir, f)
    size_mb = os.path.getsize(f_path) / (1024*1024) if os.path.isfile(f_path) else 0
    print(f" - {f} ({size_mb:.2f} MB)" if os.path.isfile(f_path) else f" - [DIR] {f}")

# Descompactar qwen3loop_tool_sft_final.zip se presente
zip_path = os.path.join(dest_dir, "qwen3loop_tool_sft_final.zip")
extract_dir = os.path.join(dest_dir, "extracted")
if os.path.exists(zip_path):
    print(f"\nDescompactando {zip_path} em {extract_dir}...", flush=True)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    print("Descompactação concluída com sucesso!", flush=True)
    
    print("\nArquivos extraídos:")
    for root, dirs, extracted_files in os.walk(extract_dir):
        for ef in extracted_files:
            p = os.path.join(root, ef)
            rel_p = os.path.relpath(p, extract_dir)
            size_mb = os.path.getsize(p) / (1024*1024)
            print(f" - {rel_p} ({size_mb:.2f} MB)")

print("\n=== PROCESSO DE DOWNLOAD E EXTRAÇÃO CONCLUÍDO COM SUCESSO! ===", flush=True)
