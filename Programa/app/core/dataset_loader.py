"""dataset_loader.py — Carregamento eficiente e indexado do manifesto e amostras."""
import json
import os
from typing import Any, Dict, List, Optional

from app.core import config, repo_state

ROOT = repo_state.ROOT


def get_manifest_path() -> str:
    p = config.get_manifest_path()
    return p if p else os.path.join(ROOT, "data", "split", "manifest.jsonl")


def get_dataset_path() -> str:
    p = config.get_dataset_path()
    return p if p else os.path.join(ROOT, "data", "cleaned", "curriculum_3000_tool_sft_PURIFIED.jsonl")


# Cache em memória para acesso instantâneo na UI
_MANIFEST_CACHE: Optional[List[Dict[str, Any]]] = None
_OFFSET_INDEX: Optional[Dict[str, int]] = None
_PURIFIED_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


def get_manifest(force_reload: bool = False) -> List[Dict[str, Any]]:
    """Carrega o manifesto completo de splits ou gera dinamicamente se não existir."""
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None and not force_reload:
        return _MANIFEST_CACHE

    records = []
    manifest_fp = get_manifest_path()
    if os.path.isfile(manifest_fp):
        with open(manifest_fp, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    else:
        # Fallback inteligente: constrói manifesto básico a partir do dataset principal
        dataset_fp = get_dataset_path()
        if os.path.isfile(dataset_fp):
            with open(dataset_fp, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        d = json.loads(line)
                        u = d.get("uuid") or d.get("id") or f"sample_{idx}"
                        records.append({
                            "uuid": u,
                            "executor": d.get("executor", "generic"),
                            "pattern": d.get("pattern", "single"),
                            "n_calls": len(d.get("tools", [])) or 1,
                            "token_len": len(str(d)) // 4,
                            "source": d.get("source", "dataset"),
                        })
    _MANIFEST_CACHE = records
    return _MANIFEST_CACHE


READ_VERBS = {
    'get', 'read', 'list', 'show', 'view', 'fetch', 'dump', 'print',
    'load', 'search', 'query', 'cat', 'ls', 'dir', 'find', 'grep',
    'head', 'tail', 'df', 'du', 'wc', 'stat', 'test', 'resolve',
    'check', 'uptime', 'location', 'timezone', 'history', 'who',
    'id', 'groups', 'diff'
}
MUTATING_VERBS = {
    'new', 'set', 'start', 'stop', 'restart', 'remove', 'delete',
    'create', 'write', 'add', 'clear', 'mkdir', 'touch', 'rm',
    'kill', 'execute', 'run', 'invoke', 'patch', 'post', 'put',
    'install', 'apt', 'pip', 'node', 'python', 'sh', 'bash'
}


def classify_effect(sample: Dict[str, Any]) -> str:
    """Classifica a amostra em 'read_only', 'mutating' ou 'none' baseado no efeito colateral."""
    calls = []
    for m in sample.get('messages', []):
        for tc in m.get('tool_calls', []):
            fn = tc.get('function', {}).get('name', '').lower()
            args_raw = tc.get('function', {}).get('arguments', '{}')
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            cmd = (args.get('command', '') or args.get('cmd', '') or '').strip().lower()
            calls.append((fn, cmd))

    if not calls:
        return 'none'

    for fn, cmd in calls:
        if cmd:
            first_token = cmd.split()[0] if cmd.split() else ''
            first_verb = first_token.split('-')[0] if '-' in first_token else first_token
            first_clean = first_verb.strip(';|\"\'')
            if first_clean in MUTATING_VERBS or any(cmd.startswith(x) for x in ['mkdir', 'touch', 'rm ', 'python ', 'node ', 'git commit', 'git push']):
                return 'mutating'
            if any(x in cmd for x in [' > ', ' >> ', ' | out-file', 'set-content', 'new-itemproperty', 'new-item']):
                return 'mutating'
        else:
            first_part = fn.split('_')[0]
            if first_part in MUTATING_VERBS or fn in ('execute_python', 'python_interpreter'):
                return 'mutating'

    return 'read_only'


def get_sample_by_uuid(uuid: str) -> Optional[Dict[str, Any]]:
    """Recupera uma amostra completa do dataset por UUID/ID."""
    global _PURIFIED_CACHE
    if _PURIFIED_CACHE is None:
        cache: Dict[str, Dict[str, Any]] = {}
        dataset_fp = get_dataset_path()
        if os.path.isfile(dataset_fp):
            with open(dataset_fp, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    if line.strip():
                        d = json.loads(line)
                        u = d.get("uuid") or d.get("id") or f"sample_{idx}"
                        d["effect"] = classify_effect(d)
                        cache[u] = d
        _PURIFIED_CACHE = cache

    return _PURIFIED_CACHE.get(uuid)


def get_taxonomy_categories() -> Dict[str, List[str]]:
    """Descobre dinamicamente os valores de categorias e taxonomia presentes no dataset."""
    manifest = get_manifest()
    executors = set()
    patterns = set()
    for m in manifest:
        if m.get("executor"):
            executors.add(m["executor"])
        if m.get("pattern"):
            patterns.add(m["pattern"])

    pillars = set()
    subpillars = set()
    if _PURIFIED_CACHE is None:
        get_sample_by_uuid("")
    if _PURIFIED_CACHE:
        for s in _PURIFIED_CACHE.values():
            if s.get("pillar"):
                pillars.add(s["pillar"])
            if s.get("subpillar"):
                subpillars.add(s["subpillar"])

    return {
        "executors": sorted(executors) if executors else ["generic", "narrow", "mixed", "abstention"],
        "patterns": sorted(patterns) if patterns else ["single", "multi", "retry", "abstention"],
        "pillars": sorted(pillars) if pillars else [
            "workspace_code", "windows_powershell", "terminal_linux",
            "python_code", "technical_api", "memory_retrieval"
        ],
        "subpillars": sorted(subpillars) if subpillars else [
            "bash", "powershell", "fs_tools", "python", "api", "memory"
        ]
    }


def query_samples(
    executor: str = "",
    pattern_calls: str = "",
    pillar: str = "",
    subpillar: str = "",
    effect: str = "",
    search_text: str = "",
    max_len: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Filtra o manifesto combinando executor, pattern, pilar, subpilar, efeito, busca textual e max_len."""
    manifest = get_manifest()
    results = []

    search_lower = search_text.strip().lower()

    for m in manifest:
        if executor and m.get("executor") != executor:
            continue

        if pattern_calls:
            pat = m.get("pattern", "")
            n_calls = m.get("n_calls", 0)
            if pattern_calls == "single" and not (pat == "single" or n_calls == 1):
                continue
            elif pattern_calls == "multi" and not (pat == "multi"):
                continue
            elif pattern_calls == "retry" and not (pat == "retry"):
                continue
            elif pattern_calls == "abstention" and not (pat == "abstention" or n_calls == 0):
                continue

        if max_len is not None and max_len > 0:
            if m.get("token_len", 0) > max_len:
                continue

        # Verifica pilar, subpilar, efeito colateral e busca textual se solicitados
        if pillar or subpillar or effect or search_lower:
            sample = get_sample_by_uuid(m["uuid"])
            if not sample:
                continue

            if pillar and sample.get("pillar") != pillar:
                continue

            if subpillar and sample.get("subpillar") != subpillar:
                continue

            if effect and sample.get("effect") != effect:
                continue

            if search_lower:
                match_uuid = search_lower in m["uuid"].lower()
                match_source = search_lower in m.get("source", "").lower()
                match_content = any(
                    search_lower in str(msg.get("content", "")).lower()
                    for msg in sample.get("messages", [])
                )
                if not (match_uuid or match_source or match_content):
                    continue

        results.append(m)

    return results
