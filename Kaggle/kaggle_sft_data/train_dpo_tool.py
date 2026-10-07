"""train_dpo_tool.py — DPO offline p/ Qwen3Loop (chosen x rejected).
Uso: python scripts/train/train_dpo_tool.py --pairs <chosen.jsonl> --rejected <rej.jsonl>
       --model_path <ckpt> --output_dir <out> [--lr 5e-7 --beta 0.1 --epochs 1]
Chosen COM mascaras (igual SFT); rejected SEM mascaras (penalizar o vicio).
"""
import argparse
import copy
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

cur_dir = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(cur_dir, "..", "..", "engine")),
          os.path.abspath(os.path.join(cur_dir, "..", ".."))):
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from qwen3loop_python import Qwen3LoopConfig, Qwen3LoopForCausalLM  # noqa: E402
AutoConfig.register("qwen3loop", Qwen3LoopConfig)
AutoModelForCausalLM.register(Qwen3LoopConfig, Qwen3LoopForCausalLM)

sys.path.insert(0, cur_dir)
from train_sft_tool_qwen3loop import (Qwen3LoopToolDataset, collate_fn,  # noqa: E402
                                      save_model_and_probe)


def seq_logps(logits, labels):
    # Alinhamento LM correto: logits[t-1] prediz labels[t].
    lp = torch.log_softmax(logits.float()[:, :-1], dim=-1)
    tgt = labels[:, 1:].clamp(min=0).unsqueeze(-1)
    out = lp.gather(-1, tgt).squeeze(-1)
    mask = (labels[:, 1:] != -100).float()
    return (out * mask).sum(-1), mask.sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--rejected", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--loop_aux", type=float, default=0.0,
                    help="peso da CE auxiliar nos chosen nos snapshots "
                         "intermediarios do loop (0=desligado)")
    ap.add_argument("--loop_aux_snaps", type=int, default=1,
                    help="quantos snapshots intermediarios (exclui o final) "
                         "supervisionar com CE")
    ap.add_argument("--sft_anchor", type=float, default=0.0,
                    help="peso da CE media nos chosen (logits finais). "
                         "Ancora de capacidade (DPO+). 0=desligado")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    pad = tok.convert_tokens_to_ids("<|endoftext|>")
    ds_c = Qwen3LoopToolDataset(args.pairs, tok, max_len=args.max_len,
                                mask_repeats=True)
    ds_r = Qwen3LoopToolDataset(args.rejected, tok, max_len=args.max_len,
                                mask_repeats=False)
    n = min(len(ds_c), len(ds_r))
    print(f"pares: {n} (chosen c/ masks, rejected sem)", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    pol = AutoModelForCausalLM.from_config(cfg).to(torch.bfloat16)
    pol.load_state_dict(__import__("safetensors.torch", fromlist=["x"]).load_file(
        os.path.join(args.model_path, "model.safetensors")), strict=False)
    pol.gradient_checkpointing_enable()
    pol.to(dev).train()
    ref = copy.deepcopy(pol).eval()
    for p in ref.parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr)
    tok_cfg = cfg

    def batch_of(ds, idxs):
        items = [ds[i] for i in idxs]
        m = max(x["input_ids"].size(0) for x in items)
        ii = torch.stack([F.pad(x["input_ids"], (0, m - x["input_ids"].size(0)),
                                value=pad) for x in items]).to(dev)
        lb = torch.stack([F.pad(x["labels"], (0, m - x["labels"].size(0)),
                                value=-100) for x in items]).to(dev)
        am = (ii != pad).long()
        return ii, lb, am

    idxs = list(range(n))
    step = 0
    pol.train()
    for ep in range(args.epochs):
        tot = 0.0
        for s in range(0, n, args.batch_size * args.grad_accum):
            chunk = idxs[s:s + args.batch_size * args.grad_accum]
            loss_acc = 0.0
            for j in range(0, len(chunk), args.batch_size):
                b = chunk[j:j + args.batch_size]
                iic, lbc, amc = batch_of(ds_c, b)
                iir, lbr, amr = batch_of(ds_r, b)
                with torch.no_grad():
                    rc = ref(input_ids=iic, attention_mask=amc).logits
                    rr = ref(input_ids=iir, attention_mask=amr).logits
                oc = pol(input_ids=iic, attention_mask=amc).logits
                orr = pol(input_ids=iir, attention_mask=amr).logits
                lpc, cntc = seq_logps(oc, lbc)
                lrc, _ = seq_logps(rc, lbc)
                lpr, _ = seq_logps(orr, lbr)
                lrr, _ = seq_logps(rr, lbr)
                margin = (lpc - lrc) - (lpr - lrr)
                loss = (-F.logsigmoid(args.beta * margin)).mean()
                if args.sft_anchor > 0:
                    ce_final = -(lpc / cntc.clamp(min=1)).mean()
                    loss = loss + args.sft_anchor * ce_final
                if args.loop_aux > 0:
                    with torch.enable_grad():
                        hid = pol.model(input_ids=iic,
                                        attention_mask=amc).loop_hidden_states
                    aux = 0.0
                    n_aux = 0
                    for s in list(hid[:-1])[-args.loop_aux_snaps:]:
                        aux_logits = pol.lm_head(
                            pol.model.norm(s.to(oc.dtype))).float()
                        lp_aux = torch.log_softmax(aux_logits[:, :-1], -1)
                        tgt = lbc[:, 1:].clamp(min=0).unsqueeze(-1)
                        ce = -lp_aux.gather(-1, tgt).squeeze(-1)
                        m = (lbc[:, 1:] != -100).float()
                        aux = aux + (ce * m).sum() / m.sum().clamp(min=1)
                        n_aux += 1
                    if n_aux:
                        loss = loss + args.loop_aux * (aux / n_aux)
                (loss / max(1, len(chunk) // args.batch_size)).backward()
                loss_acc += loss.item()
            opt.step()
            opt.zero_grad()
            step += 1
            tot += loss_acc
            if step % 5 == 0:
                print(f"step {step} loss {loss_acc:.4f}", flush=True)
        print(f"epoch {ep+1} loss-media {tot/max(step,1):.4f}", flush=True)
        ep_dir = os.path.join(args.output_dir, f"epoch-{ep+1}")
        os.makedirs(ep_dir, exist_ok=True)
        save_model_and_probe(pol, None, tok, tok_cfg, ep_dir)
        print(f"checkpoint epoca salvo em: {ep_dir}", flush=True)
    os.makedirs(args.output_dir, exist_ok=True)
    save_model_and_probe(pol, None, tok, tok_cfg, args.output_dir)
    print("DPO OK ->", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
