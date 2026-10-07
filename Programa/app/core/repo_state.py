"""repo_state.py — leitura de estado do repositorio (sem cache, sem escrita)."""
import glob
import json
import os
import subprocess

from app.core.paths import find_root

ROOT = find_root()


def _lines(fp):
    try:
        with open(os.path.join(ROOT, fp), encoding="utf-8") as f:
            return [l for l in f if l.strip()]
    except FileNotFoundError:
        return []


def dataset_n():
    return len(_lines("data/cleaned/curriculum_3000_tool_sft_PURIFIED.jsonl"))


def checkpoints():
    out = []
    ck = os.path.join(ROOT, "checkpoints")
    if os.path.isdir(ck):
        for d in sorted(os.listdir(ck)):
            p = os.path.join(ck, d)
            if os.path.isdir(p):
                st = os.path.join(p, "model.safetensors")
                out.append({"name": d, "mb": round(
                    os.path.getsize(st) / 1e6) if os.path.exists(st) else 0})
    return out


def gguf_models():
    out = []
    for fp in sorted(glob.glob(os.path.join(ROOT, "models", "gguf", "*.gguf"))):
        out.append({"name": os.path.basename(fp),
                    "mb": round(os.path.getsize(fp) / 1e6)})
    return out


def vram():
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=memory.used,memory.total",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        u, t = r.stdout.strip().split(",")
        return {"used_mb": int(u), "total_mb": int(t)}
    except Exception as e:
        return {"error": str(e)[:80]}


def servers():
    """Portas 8080-8086 com /health downhill."""
    import urllib.request
    up = []
    for p in (8080, 8081, 8082, 8083, 8084, 8085, 8086):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{p}/health",
                                        timeout=3) as r:
                up.append({"port": p,
                           "status": json.load(r).get("status", "?")})
        except Exception:
            pass
    return up


def kaggle_sessions():
    known_env = os.environ.get("KAGGLE_KERNELS")
    if known_env:
        known = [k.strip() for k in known_env.split(",") if k.strip()]
    else:
        known = ["lordnyx0/qwen3loop-tool-dpo", "lordnyx0/svg-easy-sft",
                 "lordnyx0/qwen3loop-tool-sft"]
    out = []
    try:
        token = subprocess.check_output(
            ["kaggle", "auth", "print-access-token"], text=True,
            timeout=60).strip()
    except Exception as e:
        return {"error": f"auth: {str(e)[:60]}"}
    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = token
    env.pop("KAGGLE_USERNAME", None)
    env.pop("KAGGLE_KEY", None)
    for k in known:
        try:
            r = subprocess.run(["kaggle", "kernels", "status", k], env=env,
                               capture_output=True, text=True, timeout=60)
            out.append({"kernel": k,
                        "status": (r.stdout + r.stderr).strip()[-60:]})
        except Exception as e:
            out.append({"kernel": k, "error": str(e)[:60]})
    return out


def summary():
    return {"root": ROOT, "purified_n": dataset_n(),
            "checkpoints": checkpoints(), "gguf": gguf_models(),
            "vram": vram(), "servers": servers()}
