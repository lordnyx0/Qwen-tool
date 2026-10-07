import os
import shutil

ROOT = os.path.abspath(".")
SCRATCH = os.path.join(ROOT, "scratch")
LOGS = os.path.join(ROOT, "logs")

# 1. Organizar scratch/ em subdiretorios tematicos
groups = {
    "monitoring": [
        "monitor_v13.py", "monitor_v14.py", "monitor_v15.py", "monitor_v16.py",
        "poll_v12.py", "stream_sft_live.py"
    ],
    "tests": [
        "test_checkpoint_equivalence.py", "test_chunked_loss_equivalence.py",
        "test_batch1_max_len.py", "test_abstain_generation.py", "test_l1_generation.py",
        "test_l3_max512.py", "test_l3_perf.py", "test_loop_checkpointing_vram.py",
        "test_single_generation.py", "test_single_random.py", "test_extract.py",
        "fast_test_batch1.py", "validate_end_to_end_training.py", "verify_train_step.py",
        "inspect_sdpa.py", "sdpa_src.py"
    ],
    "audits": [
        "audit_50_samples_per_level.py", "deep_dataset_audit.py", "check_dataset_alignment.py",
        "check_v11_steps.py", "check_v12_steps.py", "inspect_eval_results.py",
        "inspect_transcript.py"
    ],
    "deploy": [
        "push_kaggle.py", "push_kernel.py", "patch_notebook.py", "rebuild_notebook_clean.py",
        "update_nb_params.py", "update_notebook_6k.py", "deploy_true_sdpa.py",
        "filter_clean_dataset.py", "analyze_max_len_impact.py", "upload_base_sft_to_hf.py",
        "build_clean_curated_sft_dataset.py"
    ]
}

for folder, files in groups.items():
    dest_dir = os.path.join(SCRATCH, folder)
    os.makedirs(dest_dir, exist_ok=True)
    for fname in files:
        src = os.path.join(SCRATCH, fname)
        if os.path.exists(src):
            dst = os.path.join(dest_dir, fname)
            shutil.move(src, dst)
            print(f"scratch -> {folder}/{fname}")

# 2. Organizar logs/ antigos em logs/kaggle_runs/
old_logs = [
    "kaggle_v11_run.log", "kaggle_v12_full.log", "kaggle_v12_run.log",
    "kaggle_v13_live.log", "kaggle_v14_live.log", "kaggle_v15_live.log",
    "kaggle_v16_live.log", "sft_kernel_run.log", "sft_kernel_run_v2.log",
    "sft_kernel_run_v3.log", "sft_kernel_run_v4.log", "sft_kernel_run_v4_complete.log",
    "sft_live_stream.log", "kaggle_full.log"
]

runs_dir = os.path.join(LOGS, "kaggle_runs")
os.makedirs(runs_dir, exist_ok=True)

for l in old_logs:
    src = os.path.join(LOGS, l)
    if os.path.exists(src):
        dst = os.path.join(runs_dir, l)
        shutil.move(src, dst)
        print(f"logs -> kaggle_runs/{l}")

# 3. Limpeza de __pycache__ recursivo
for root, dirs, files in os.walk(ROOT):
    for d in dirs:
        if d == "__pycache__":
            p = os.path.join(root, d)
            try:
                shutil.rmtree(p)
                print(f"Limpo cache: {p}")
            except Exception:
                pass

print("\nOrganização da codebase concluída com sucesso!")
