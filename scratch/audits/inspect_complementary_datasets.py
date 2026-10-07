import sys
import json
from datasets import load_dataset

sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("INSPECIONANDO DATASETS COMPLEMENTARES DE ELITE")
print("=" * 70)

# 1. Inspecionar Glaive Function Calling v2
print("\n1. Conectando a glaiveai/glaive-function-calling-v2...")
try:
    ds_glaive = load_dataset("glaiveai/glaive-function-calling-v2", split="train", streaming=True)
    count = 0
    negatives = 0
    positives = 0
    for row in ds_glaive:
        count += 1
        chat = row.get("chat", "")
        if "<functioncall>" in chat:
            positives += 1
        else:
            negatives += 1
        if count == 1:
            print("--- Exemplo Positivo Glaive ---")
            print("System:", row.get("system")[:150])
            print("Chat:", row.get("chat")[:200])
        if negatives == 1 and "<functioncall>" not in chat:
            print("\n--- Exemplo Negativo (Abstenção) Glaive ---")
            print("System:", row.get("system")[:150])
            print("Chat:", row.get("chat")[:200])
        if count >= 50:
            break
    print(f"Glaive (primeiras 50): {positives} positivas, {negatives} abstenções negativas")
except Exception as e:
    print(f"Erro Glaive: {e}")

# 2. Inspecionar Salesforce/xlam-function-calling-60k
print("\n2. Conectando a Salesforce/xlam-function-calling-60k...")
try:
    ds_xlam = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True)
    for i, row in enumerate(ds_xlam):
        if i >= 2: break
        print(f"--- Exemplo {i+1} xLAM ---")
        print("Keys:", row.keys())
        print("Query:", row.get("query")[:150] if "query" in row else row.get("instruction")[:150])
        tools = row.get("tools")
        print("Tools type/sample:", type(tools), str(tools)[:150])
        answers = row.get("answers")
        print("Answers:", str(answers)[:150])
except Exception as e:
    print(f"Erro xLAM: {e}")
