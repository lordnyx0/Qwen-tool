import json

nb_path = "kaggle_kernel/qwen3loop_tool_sft.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        new_source = []
        for line in source:
            if "--max_len" in line:
                # Replace whatever max_len is with 4608
                import re
                line = re.sub(r"--max_len\s+\d+", "--max_len 4608", line)
            new_source.append(line)
        cell["source"] = new_source

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook atualizado para --max_len 4608 com sucesso!")
