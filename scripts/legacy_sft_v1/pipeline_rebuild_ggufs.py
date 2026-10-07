"""
pipeline_rebuild_ggufs.py  --  recomeça no Step 4 (F16 ja existem).
"""
import sys, subprocess
from pathlib import Path

REPO_ID      = "Lordnyx/qwen3loop-0.6b-sft-deep-supervision-v1"
WORK_DIR     = Path(r"C:\Users\Nyx\Downloads\Diversos\qwen")
F16_GGUF     = WORK_DIR / "modelo_qwen3loop_sft_f16.gguf"
Q8_GGUF      = WORK_DIR / "modelo_qwen3loop_sft_q8_0.gguf"
UNROLLED_F16 = WORK_DIR / "unrolled_modelo_qwen3loop_sft_f16.gguf"
UNROLLED_Q8  = WORK_DIR / "unrolled_modelo_qwen3loop_sft_q8_0.gguf"
QUANT_SCRIPT = WORK_DIR / "quantize_q8_0.py"
PYTHON       = sys.executable
HF_CLI       = "hf"

def run(cmd):
    print(f"\n>>> {' '.join(str(c) for c in cmd)}\n")
    subprocess.run([str(c) for c in cmd], check=True)

def step(n, t):
    print(f"\n{'='*65}\n  STEP {n}: {t}\n{'='*65}")

# ── STEP 4: Quantize ──────────────────────────────────────────────
step(4, "Quantize F16 -> Q8_0 (Python quantizer)")

if Q8_GGUF.exists():
    print(f"[SKIP] {Q8_GGUF.name} ({Q8_GGUF.stat().st_size/1e9:.2f} GB)")
else:
    run([PYTHON, QUANT_SCRIPT, str(F16_GGUF), str(Q8_GGUF)])

if UNROLLED_Q8.exists():
    print(f"[SKIP] {UNROLLED_Q8.name} ({UNROLLED_Q8.stat().st_size/1e9:.2f} GB)")
else:
    run([PYTHON, QUANT_SCRIPT, str(UNROLLED_F16), str(UNROLLED_Q8)])

print(f"\nArquivos gerados:")
for f in [F16_GGUF, Q8_GGUF, UNROLLED_F16, UNROLLED_Q8]:
    print(f"  {f.name:50s} {f.stat().st_size/1e9:.2f} GB")

# ── STEP 5: Upload ────────────────────────────────────────────────
step(5, "Upload GGUFs para HuggingFace")
for local, hf_path in [
    (F16_GGUF,     F16_GGUF.name),
    (Q8_GGUF,      Q8_GGUF.name),
    (UNROLLED_F16, UNROLLED_F16.name),
    (UNROLLED_Q8,  UNROLLED_Q8.name),
]:
    print(f"\n  {local.name} -> {REPO_ID}/{hf_path}")
    run([HF_CLI, "upload", REPO_ID, str(local), hf_path])

print(f"\n{'='*65}")
print("  PIPELINE CONCLUIDO!")
print(f"  https://huggingface.co/{REPO_ID}")
print(f"{'='*65}\n")
