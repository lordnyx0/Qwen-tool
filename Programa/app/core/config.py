"""config.py — Gerenciamento declarativo e auto-descoberta de configuração do Workbench."""
import glob
import json
import os
from typing import Any, Dict, List, Optional

from app.core.paths import find_root

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def get_project_config(force_reload: bool = False) -> Dict[str, Any]:
    """Retorna a configuração do projeto atual, com auto-descoberta inteligente."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    root = find_root()
    cfg_file = os.path.join(root, "workbench.json")
    if not os.path.isfile(cfg_file):
        cfg_file = os.path.join(root, "project_config.json")

    # Layout e valores padrão universais
    folder_name = os.path.basename(root).replace("_", " ").title()
    config: Dict[str, Any] = {
        "project_name": "Qwen Control Center" if "qwen" in folder_name.lower() else f"{folder_name} Workbench",
        "dataset_path": "",
        "manifest_path": "",
        "id_field": "uuid",
        "content_field": "messages",
        "taxonomy": {},
        "jobs": {},
        "evals": {},
    }

    # Carrega arquivo declarativo se existir
    if os.path.isfile(cfg_file):
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
                config.update(user_cfg)
        except Exception:
            pass

    # 1. Auto-descoberta de dataset principal (.jsonl)
    if not config["dataset_path"]:
        default_purified = os.path.join(root, "data", "cleaned", "curriculum_3000_tool_sft_PURIFIED.jsonl")
        if os.path.isfile(default_purified):
            config["dataset_path"] = default_purified
        else:
            candidates = glob.glob(os.path.join(root, "data", "**", "*.jsonl"), recursive=True)
            non_manifest = [c for c in candidates if "manifest" not in os.path.basename(c).lower()]
            if non_manifest:
                config["dataset_path"] = os.path.abspath(non_manifest[0])
            elif candidates:
                config["dataset_path"] = os.path.abspath(candidates[0])

    # 2. Auto-descoberta de manifesto (.jsonl)
    if not config["manifest_path"]:
        default_manifest = os.path.join(root, "data", "split", "manifest.jsonl")
        if os.path.isfile(default_manifest):
            config["manifest_path"] = default_manifest
        else:
            m_cands = glob.glob(os.path.join(root, "data", "**", "*manifest*.jsonl"), recursive=True)
            if m_cands:
                config["manifest_path"] = os.path.abspath(m_cands[0])

    _CONFIG_CACHE = config
    return _CONFIG_CACHE


def get_dataset_path() -> str:
    """Retorna o caminho absoluto do dataset atual."""
    return get_project_config().get("dataset_path", "")


def get_manifest_path() -> str:
    """Retorna o caminho do manifesto atual."""
    return get_project_config().get("manifest_path", "")


def get_project_name() -> str:
    """Retorna o nome do projeto."""
    return get_project_config().get("project_name", "Workbench")
