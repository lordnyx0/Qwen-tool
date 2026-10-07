"""
upload_all_to_hf.py
Faz o upload de todos os componentes atualizados para:
Lordnyx/qwen3loop-0.6b-sft-deep-supervision-v1
"""

import sys
import time
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "Lordnyx/qwen3loop-0.6b-sft-deep-supervision-v1"
WORK_DIR = Path(r"C:\Users\Nyx\Downloads\Diversos\qwen")
CKPT_DIR = WORK_DIR / "checkpoints" / "qwen3loop_sft_final"

api = HfApi()

UPLOADS = [
    # Metadados e Model Card
    {
        "local": WORK_DIR / "README_hf_updated.md",
        "remote": "README.md",
        "desc": "Model Card (README.md) atualizado com métricas SFT e benchmarks"
    },
    # Configurações e Tokenizer
    {
        "local": CKPT_DIR / "config.json",
        "remote": "config.json",
        "desc": "config.json atualizado"
    },
    {
        "local": CKPT_DIR / "tokenizer.json",
        "remote": "tokenizer.json",
        "desc": "tokenizer.json"
    },
    {
        "local": CKPT_DIR / "tokenizer_config.json",
        "remote": "tokenizer_config.json",
        "desc": "tokenizer_config.json"
    },
    {
        "local": CKPT_DIR / "chat_template.jinja",
        "remote": "chat_template.jinja",
        "desc": "chat_template.jinja"
    },
    # Engine Python corrigida
    {
        "local": WORK_DIR / "engine" / "qwen3loop_python" / "modeling_qwen3loop.py",
        "remote": "engine/qwen3loop_python/modeling_qwen3loop.py",
        "desc": "modeling_qwen3loop.py com suporte corrigido a loop_hidden_states"
    },
    # Sonda Halting Probe (MLP)
    {
        "local": CKPT_DIR / "latent_halting_probe.pt",
        "remote": "latent_halting_probe.pt",
        "desc": "Sonda MLP Latente (latent_halting_probe.pt - 91.5% acc)"
    },
    # Pesos PyTorch Safetensors
    {
        "local": CKPT_DIR / "model.safetensors",
        "remote": "model.safetensors",
        "desc": "model.safetensors (Pesos finais do SFT - 1.14 GB)"
    },
    # Modelos GGUF Q8_0 e F16
    {
        "local": WORK_DIR / "unrolled_modelo_qwen3loop_sft_q8_0.gguf",
        "remote": "unrolled_modelo_qwen3loop_sft_q8_0.gguf",
        "desc": "unrolled_modelo_qwen3loop_sft_q8_0.gguf (56 camadas, Q8_0 - 1.11 GB)"
    },
    {
        "local": WORK_DIR / "modelo_qwen3loop_sft_q8_0.gguf",
        "remote": "modelo_qwen3loop_sft_q8_0.gguf",
        "desc": "modelo_qwen3loop_sft_q8_0.gguf (28 blocos nativos, Q8_0 - 0.64 GB)"
    },
    {
        "local": WORK_DIR / "unrolled_modelo_qwen3loop_sft_f16.gguf",
        "remote": "unrolled_modelo_qwen3loop_sft_f16.gguf",
        "desc": "unrolled_modelo_qwen3loop_sft_f16.gguf (56 camadas, FP16 - 2.08 GB)"
    },
    {
        "local": WORK_DIR / "modelo_qwen3loop_sft_f16.gguf",
        "remote": "modelo_qwen3loop_sft_f16.gguf",
        "desc": "modelo_qwen3loop_sft_f16.gguf (28 blocos nativos, FP16 - 1.20 GB)"
    },
]

def main():
    print("=" * 70)
    print(f" UPLOAD PARA O HUGGING FACE: {REPO_ID}")
    print("=" * 70)

    user = api.whoami()
    print(f"Autenticado como: {user['name']}\n")

    total_files = len(UPLOADS)
    for idx, item in enumerate(UPLOADS, 1):
        loc = item["local"]
        rem = item["remote"]
        desc = item["desc"]

        if not loc.exists():
            print(f"[{idx}/{total_files}] [ERRO] Arquivo não encontrado: {loc}")
            continue

        size_mb = loc.stat().st_size / (1024 * 1024)
        print(f"[{idx}/{total_files}] Enviando: {rem} ({size_mb:.2f} MB)")
        print(f"       Descrição: {desc}")

        t0 = time.time()
        try:
            api.upload_file(
                path_or_fileobj=str(loc),
                path_in_repo=rem,
                repo_id=REPO_ID,
                commit_message=f"Update {rem}: {desc}"
            )
            dt = time.time() - t0
            speed = size_mb / dt if dt > 0 else 0
            print(f"       [SUCESSO] Concluído em {dt:.1f}s ({speed:.2f} MB/s)\n")
        except Exception as e:
            print(f"       [FALHA] Erro no upload: {e}\n")

    print("=" * 70)
    print(" TODOS OS UPLOADS FINALIZADOS!")
    print(f" Verifique o repositório em: https://huggingface.co/{REPO_ID}")
    print("=" * 70)

if __name__ == "__main__":
    main()
