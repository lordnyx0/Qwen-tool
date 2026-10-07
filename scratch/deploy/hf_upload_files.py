"""Upload por arquivo (progressivo, retomavel): pula o que ja existe no repo."""
from huggingface_hub import HfApi, list_repo_files

REPO = "Lordnyx/qwen3loop-0.6b-tool-calling-sft-v1"
FOLDER = "hf_tool_sft_stage"
FILES = [
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "latent_halting_probe.pt",
    "engine/README.md",
    "engine/qwen3loop_python/__init__.py",
    "engine/qwen3loop_python/configuration_qwen3loop.py",
    "engine/qwen3loop_python/modeling_qwen3loop.py",
    "engine/qwen3loop_python/modeling_qwen3loop_minimal.py",
    "engine/llamacpp_patch/README.md",
    "engine/llamacpp_patch/src/llama-model-loader.cpp",
    "engine/llamacpp_patch/src/models/qwen35.cpp",
    "modelo_qwen3loop_tool_sft_q8_0.gguf",
    "modelo_qwen3loop_tool_sft_f16.gguf",
    "unrolled_modelo_qwen3loop_tool_sft_q8_0.gguf",
    "unrolled_modelo_qwen3loop_tool_sft_f16.gguf",
    "model.safetensors",
]
api = HfApi()
have = set(list_repo_files(REPO))
print(f"no repo: {len(have)}", flush=True)
for f in FILES:
    if f in have:
        print("skip (existe):", f, flush=True)
        continue
    print("upando:", f, flush=True)
    api.upload_file(path_or_fileobj=f"{FOLDER}/{f}", path_in_repo=f,
                    repo_id=REPO, repo_type="model")
    print("ok:", f, flush=True)
print("UPLOAD COMPLETO", flush=True)
