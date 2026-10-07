import json
import re
from datasets import load_dataset

print("Carregando amostras do Hermes...")
ds_ht = load_dataset("sroecker/hermes-agent-traces-chatml", split="train")

parsed_ok = 0
parsed_fail = 0

for i in range(min(100, len(ds_ht))):
    item = ds_ht[i]
    msgs = item.get("messages", [])
    for m in msgs:
        content = m.get("content", "")
        if "<tool_call>" in content and "</tool_call>" in content:
            # Extrair blocos
            matches = re.findall(r"<tool_call>(.*?)</tool_call>", content, re.DOTALL)
            for raw_tc in matches:
                raw_tc = raw_tc.strip()
                try:
                    tc_obj = json.loads(raw_tc)
                    parsed_ok += 1
                except Exception as e:
                    parsed_fail += 1

print(f"Resultados Hermes (primeiras 100): Sucesso={parsed_ok} | Falha={parsed_fail}")
