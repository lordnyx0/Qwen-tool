import json, os, glob
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

tok = AutoTokenizer.from_pretrained('checkpoints/qwen3loop_tool_sft_final')

def print_bucket_stats(name, toks):
    s = pd.Series(toks)
    b1 = (s < 1000).sum()
    b2 = ((s >= 1000) & (s < 2000)).sum()
    b3 = ((s >= 2000) & (s < 3000)).sum()
    b4 = ((s >= 3000) & (s < 4000)).sum()
    b5 = ((s >= 4000) & (s <= 5000)).sum()
    over5k = (s > 5000).sum()
    total = len(s)
    print(f'=== {name} (Sampled/Total: {total}) ===')
    print(f'  Min: {s.min()} | 25%: {s.quantile(0.25):.0f} | Median: {s.median():.0f} | Mean: {s.mean():.0f} | 75%: {s.quantile(0.75):.0f} | Max: {s.max()}')
    print(f'  <1000:       {b1:5d} ({b1/total*100:5.1f}%)')
    print(f'  1000 - 1999: {b2:5d} ({b2/total*100:5.1f}%)')
    print(f'  2000 - 2999: {b3:5d} ({b3/total*100:5.1f}%)')
    print(f'  3000 - 3999: {b4:5d} ({b4/total*100:5.1f}%)')
    print(f'  4000 - 5000: {b5:5d} ({b5/total*100:5.1f}%)')
    if over5k > 0:
        print(f'  >5000:       {over5k:5d} ({over5k/total*100:5.1f}%)')
    print()

# 1. PSStrikes-commands
print('Measuring PSStrikes-commands...', flush=True)
p_ps = hf_hub_download(repo_id='dessertlab/PSStrikes-commands', filename='data/train-00000-of-00001-25809ad170aa95b5.parquet', repo_type='dataset')
df_ps = pd.read_parquet(p_ps)
ps_toks = []
for idx, r in df_ps.iterrows():
    text = f"User: {r['nl']}\nAssistant: {r['code']}"
    ps_toks.append(len(tok.encode(text)))
print_bucket_stats('PSStrikes PowerShell Commands', ps_toks)

# 2. Hermes Agent Traces (Sample 300)
print('Measuring Hermes Agent Traces...', flush=True)
ds_ht = load_dataset('sroecker/hermes-agent-traces-chatml', split='train')
ht_toks = []
for i in range(300):
    msgs = ds_ht[i].get('messages', [])
    text = ' '.join(str(m.get('content', '')) for m in msgs)
    # estimate token length fast or encode
    if len(text) > 30000:
        ht_toks.append(len(text) // 3) # rough estimate for massive outliers
    else:
        ht_toks.append(len(tok.encode(text)))
print_bucket_stats('Hermes Agent Traces ChatML (sroecker)', ht_toks)

# 3. Nemotron-Terminal-Corpus
print('Measuring Nemotron-Terminal-Corpus...', flush=True)
p_nemo = hf_hub_download(repo_id='nvidia/Nemotron-Terminal-Corpus', filename='synthetic_tasks/skill_based/easy/file_operations/data_filtered.parquet', repo_type='dataset')
df_nemo = pd.read_parquet(p_nemo)
nemo_toks = []
for i in range(150):
    convs = df_nemo.iloc[i]['conversations']
    text = ' '.join(str(m.get('content', '')) for m in convs)
    if len(text) > 35000:
        nemo_toks.append(len(text) // 3)
    else:
        nemo_toks.append(len(tok.encode(text)))
print_bucket_stats('NVIDIA Nemotron-Terminal-Corpus', nemo_toks)
