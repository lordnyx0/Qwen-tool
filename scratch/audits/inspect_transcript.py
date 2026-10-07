import json

log_file = r'C:\Users\Nyx\.gemini\antigravity-ide\brain\ff8394ef-b93f-4959-86bf-35cec41fbf4d\.system_generated\logs\transcript.jsonl'
with open(log_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if 'kaggle ' in line:
            d = json.loads(line)
            txt = str(d.get('content', ''))
            calls = [str(c.get('args')) for c in d.get('tool_calls', []) if 'kaggle' in str(c)]
            if calls:
                print(f"Call at line {i}: {calls[:1]}")
            elif 'kaggle' in txt and ('command' in txt.lower() or 'upload' in txt.lower()):
                print(f"Text at line {i}: {txt[:120]}")
