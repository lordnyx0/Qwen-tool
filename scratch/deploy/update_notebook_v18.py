import json
import re

nb_path = "kaggle_kernel/qwen3loop_tool_sft.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        new_source = []
        for line in source:
            if "--max_len" in line:
                line = re.sub(r"--max_len\s+\d+", "--max_len 5000", line)
            elif "--epochs" in line:
                line = re.sub(r"--epochs\s+\d+", "--epochs 1", line)
            elif "--eval_steps" in line:
                line = re.sub(r"--eval_steps\s+\d+", "--eval_steps 25", line)
            elif "--save_steps" in line:
                line = re.sub(r"--save_steps\s+\d+", "--save_steps 35", line)
            new_source.append(line)
        cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook atualizado para v18 (--epochs 1, --max_len 5000, --save_steps 35, --eval_steps 25) com sucesso!")
