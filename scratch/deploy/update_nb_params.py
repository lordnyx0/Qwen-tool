import json

nb_path = 'kaggle_kernel/qwen3loop_tool_sft.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    source = cell.get('source', [])
    found = False
    for line in source:
        if 'accelerate launch' in line:
            found = True
            break
    if found:
        cell['source'] = [
            "# 4. Executar Treinamento Full-Parameter SFT em Dual GPU Tesla T4 (Accelerate DDP)\n",
            "output_dir = '/kaggle/working/qwen3loop_tool_sft_final'\n",
            "\n",
            "!export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && accelerate launch \\\n",
            "    --multi_gpu \\\n",
            "    --num_processes 2 \\\n",
            "    --num_machines 1 \\\n",
            "    --mixed_precision no \\\n",
            "    --dynamo_backend no \\\n",
            "    {script_path} \\\n",
            "    --model_path {model_path} \\\n",
            "    --data_path {data_path} \\\n",
            "    --output_dir {output_dir} \\\n",
            "    --batch_size 1 \\\n",
            "    --grad_accum 8 \\\n",
            "    --epochs 2 \\\n",
            "    --lr 1.5e-5 \\\n",
            "    --max_len 4096 \\\n",
            "    --log_steps 1 \\\n",
            "    --eval_steps 30 \\\n",
            "    --save_steps 90\n"
        ]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Notebook updated with --mixed_precision no successfully.')
