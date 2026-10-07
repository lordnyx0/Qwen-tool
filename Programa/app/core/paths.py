"""paths.py — localiza a raiz do projeto (fonte ou executável) e resolve caminhos dinâmicos."""
import glob
import os
import sys
import tempfile
from pathlib import Path


def _is_valid_root(path: str) -> bool:
    """Verifica se um diretório é uma raiz válida de projeto."""
    if not path or not os.path.isdir(path):
        return False
    # Marcador declarativo explícito
    if os.path.isfile(os.path.join(path, "workbench.json")):
        return True
    if os.path.isfile(os.path.join(path, "project_config.json")):
        return True
    # Marcador de dataset específico do Qwen
    if os.path.isfile(os.path.join(path, "data", "cleaned", "curriculum_3000_tool_sft_PURIFIED.jsonl")):
        return True
    # Marcador genérico de repositório
    if os.path.isdir(os.path.join(path, ".git")):
        return True
    # Estrutura típica data/ + scripts/
    if os.path.isdir(os.path.join(path, "data")) and os.path.isdir(os.path.join(path, "scripts")):
        return True
    return False


def find_root() -> str:
    """Localiza a raiz do projeto atual sem depender de caminhos rígidos."""
    # 1. Variáveis de ambiente explícitas vencem tudo
    for env_var in ("WORKBENCH_PROJECT_ROOT", "PROJECT_ROOT", "QWEN_ROOT"):
        env_val = os.environ.get(env_var)
        if env_val and os.path.isdir(env_val):
            return os.path.abspath(env_val)

    # 2. Candidatos a partir do executável ou arquivo de script
    cands = []
    if getattr(sys, "frozen", False):
        cands.append(os.path.dirname(sys.executable))
        cands.append(os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "..")))
    
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    cands.append(os.path.abspath(os.path.join(current_file_dir, "..", "..", "..")))
    cands.append(os.path.abspath(os.path.join(current_file_dir, "..", "..")))
    cands.append(os.getcwd())

    # Procura subindo até 4 níveis em cada candidato
    for c in cands:
        curr = c
        for _ in range(4):
            if _is_valid_root(curr):
                return os.path.abspath(curr)
            parent = os.path.dirname(curr)
            if parent == curr:
                break
            curr = parent

    # 3. Fallback: layout padrão (Programa/app/core -> raiz)
    return os.path.abspath(os.path.join(current_file_dir, "..", "..", ".."))


def get_log_dir() -> str:
    """Retorna diretório de logs e temporários do projeto de forma portável."""
    root = find_root()
    # Preferência 1: <ROOT>/.cache/logs
    local_log_dir = os.path.join(root, ".cache", "logs")
    try:
        os.makedirs(local_log_dir, exist_ok=True)
        # Testa escrita
        test_file = os.path.join(local_log_dir, ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
        return os.path.abspath(local_log_dir)
    except Exception:
        pass

    # Preferência 2: Temp do sistema operacional
    fallback = os.path.join(tempfile.gettempdir(), "workbench_logs")
    os.makedirs(fallback, exist_ok=True)
    return os.path.abspath(fallback)


def find_hf_model_snapshot(repo_id: str) -> str:
    """Localiza o snapshot de um modelo HuggingFace no cache local sem caminhos de usuário hardcoded."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        hub_dir = os.path.join(hf_home, "hub")
    else:
        hub_dir = os.path.join(Path.home(), ".cache", "huggingface", "hub")

    # Transforma "Lordnyx/model-name" em "models--Lordnyx--model-name"
    parts = repo_id.strip("/").split("/")
    model_folder = f"models--{'--'.join(parts)}"
    snapshots_dir = os.path.join(hub_dir, model_folder, "snapshots")

    if os.path.isdir(snapshots_dir):
        subdirs = sorted(
            [os.path.join(snapshots_dir, d) for d in os.listdir(snapshots_dir)],
            key=os.path.getmtime,
            reverse=True
        )
        if subdirs and os.path.isdir(subdirs[0]):
            return subdirs[0].replace("\\", "/")

    return ""

