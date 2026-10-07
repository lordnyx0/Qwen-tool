"""Download do output de kernel via kagglesdk com OAuth (bypass do CLI).
Uso: python kaggle_kernel_output_dl.py [OUT_DIR] [KERNEL_SLUG]
Baseado em kaggle_output_dl.py (SFT v21, provado)."""
import json, os, sys, time, urllib.request, urllib.parse

CREDS = "C:/Users/Nyx/.kaggle/credentials.json"
OUT = sys.argv[1] if len(sys.argv) > 1 else "kaggle_kernel_output/grpo_v4"
OWNER = "lordnyx0"
SLUG = sys.argv[2] if len(sys.argv) > 2 else "qwen3loop-tool-grpo-v3-train"
VERSION = sys.argv[3] if len(sys.argv) > 3 else ""
FILTER = sys.argv[4] if len(sys.argv) > 4 else ""

def load_creds():
    with open(CREDS) as f:
        return json.load(f)

def refresh(creds):
    data = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": creds["refresh_token"],
        "client_id": "kagglesdk",
    }).encode()
    req = urllib.request.Request("https://www.kaggle.com/api/v1/oauth2/token", data=data)
    with urllib.request.urlopen(req, timeout=60) as r:
        tok = json.load(r)
    creds["access_token"] = tok["access_token"]
    if "refresh_token" in tok:
        creds["refresh_token"] = tok["refresh_token"]
    with open(CREDS, "w") as f:
        json.dump(creds, f)
    return creds

def main():
    os.makedirs(OUT, exist_ok=True)
    creds = load_creds()
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.kaggle_env import KaggleEnv
    from kagglesdk.kernels.services.kernels_api_service import (
        ApiListKernelSessionOutputRequest)

    def client(tok):
        return KaggleClient(env=KaggleEnv.PROD, api_token=tok)

    def try_list(tok):
        with client(tok) as kaggle:
            out = []
            page = ""
            while True:
                req = ApiListKernelSessionOutputRequest()
                req.user_name = OWNER
                req.kernel_slug = SLUG
                req.page_size = 100
                if VERSION:
                    req.version_label = VERSION
                if page:
                    req.page_token = page
                resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
                out.extend(resp.files or [])
                page = getattr(resp, "next_page_token", "") or ""
                if not page:
                    break
            resp.files = out
            return resp

    try:
        resp = try_list(creds["access_token"])
    except Exception as e:
        print("list falhou, renovando token:", str(e)[:200])
        creds = refresh(creds)
        resp = try_list(creds["access_token"])

    files = [(f.file_name, f.url) for f in (resp.files or [])]
    if FILTER:
        files = [(n, u) for n, u in files if n.startswith(FILTER)]
    print(f"arquivos no output: {len(files)}")
    for name, url in files:
        print(" -", name)

    with open(os.path.join(OUT, "_output_manifest.json"), "w") as f:
        json.dump([{"file_name": n, "url": u} for n, u in files], f, indent=1)

    H = {"Authorization": "Bearer " + creds["access_token"]}
    for name, url in files:
        dest = os.path.join(OUT, name)
        if os.path.exists(dest):
            print("skip (existe):", name)
            continue
        dl = f"https://www.kaggle.com/api/v1/kernels/output/download/{OWNER}/{SLUG}/{name}"
        req = urllib.request.Request(dl, headers=H)
        t0 = time.time()
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        try:
            with urllib.request.urlopen(req, timeout=1200) as r, open(dest, "wb") as f:
                total = 0
                while True:
                    chunk = r.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
            print(f"baixado: {name} ({total/1e6:.1f} MB, {time.time()-t0:.0f}s)")
        except Exception as e:
            print(f"FALHA (pulando): {name}: {str(e)[:150]}")
            try:
                os.remove(dest)
            except Exception:
                pass

if __name__ == "__main__":
    sys.exit(main())
