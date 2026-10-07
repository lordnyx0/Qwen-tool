"""
build_f16_base_from_safetensors.py (STEP 0 do pipeline GGUF)

Gera o GGUF F16 base de 28 blocos (template que o passo 1 do
`build_and_quantize_tool_sft_gguf.py` espera em
`models/gguf/modelo_qwen3loop_sft_f16.gguf`).

- Tensores: lidos do `model.safetensors` (pesos reais, F16; norms em F32).
- Metadados KV + tokenizer: copiados de um GGUF de referência (ex: Q8 antigo),
  com `block_count` ajustado para 28.

Uso:
    python scripts/quantize/build_f16_base_from_safetensors.py \
        --st_dir <dir_com_model.safetensors> \
        --ref_gguf models/gguf/legacy_v1/unrolled_modelo_qwen3loop_tool_sft_q8_0.gguf
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import safetensors.torch

try:
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
except ImportError:
    print("ERRO: pacote gguf nao encontrado. Instale com: pip install gguf")
    sys.exit(1)

ROOT_DIR = Path(os.path.abspath(".")).resolve()
DEFAULT_OUT = ROOT_DIR / "models" / "gguf" / "modelo_qwen3loop_sft_f16.gguf"

ST_TO_GGUF = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
LAYER_MAP = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
ST_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def st_to_gguf_name(st_name: str):
    if st_name in ST_TO_GGUF:
        return ST_TO_GGUF[st_name]
    m = ST_RE.match(st_name)
    if m and m.group(2) in LAYER_MAP:
        return f"blk.{int(m.group(1))}.{LAYER_MAP[m.group(2)]}"
    return None


def copy_kv(reader, writer, arch: str, n_blocks: int):
    skip = {
        "general.architecture", "general.file_type", "general.quantization_version",
        "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
        f"{arch}.block_count",
    }
    for name, field in reader.fields.items():
        if name in skip:
            continue
        try:
            ftype = field.types[0]
            val = field.parts[-1]
            if ftype == GGUFValueType.STRING:
                writer.add_string(name, bytes(val).decode("utf-8").rstrip("\x00"))
            elif ftype == GGUFValueType.UINT32: writer.add_uint32(name, int(val[0]))
            elif ftype == GGUFValueType.INT32: writer.add_int32(name, int(val[0]))
            elif ftype == GGUFValueType.FLOAT32: writer.add_float32(name, float(val[0]))
            elif ftype == GGUFValueType.UINT64: writer.add_uint64(name, int(val[0]))
            elif ftype == GGUFValueType.INT64: writer.add_int64(name, int(val[0]))
            elif ftype == GGUFValueType.BOOL: writer.add_bool(name, bool(val[0]))
            elif ftype == GGUFValueType.FLOAT64: writer.add_float64(name, float(val[0]))
            elif ftype == GGUFValueType.UINT8: writer.add_uint8(name, int(val[0]))
            elif ftype == GGUFValueType.INT8: writer.add_int8(name, int(val[0]))
            elif ftype == GGUFValueType.UINT16: writer.add_uint16(name, int(val[0]))
            elif ftype == GGUFValueType.INT16: writer.add_int16(name, int(val[0]))
            elif ftype == GGUFValueType.ARRAY:
                elem = field.types[1]
                if elem == GGUFValueType.STRING:
                    writer.add_array(name, [bytes(p) for p in field.parts[6::2]])
                elif elem == GGUFValueType.INT32:
                    writer.add_array(name, [int(p[0]) for p in field.parts[5:]])
                elif elem == GGUFValueType.UINT32:
                    writer.add_array(name, [int(p[0]) for p in field.parts[5:]])
                elif elem == GGUFValueType.FLOAT32:
                    writer.add_array(name, [float(p[0]) for p in field.parts[5:]])
        except Exception:
            pass
    writer.add_uint32(f"{arch}.block_count", n_blocks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st_dir", required=True)
    ap.add_argument("--ref_gguf", required=True)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    t0 = time.time()
    st_path = str(Path(args.st_dir) / "model.safetensors")
    print(f"Lendo safetensors: {st_path}")
    st = safetensors.torch.load_file(st_path)

    gguf_names = {}
    max_blk = -1
    for k in st.keys():
        g = st_to_gguf_name(k)
        if g is None:
            print(f"AVISO: tensor sem mapeamento GGUF: {k}")
            continue
        gguf_names[k] = g
        m = re.match(r"^blk\.(\d+)\.", g)
        if m:
            max_blk = max(max_blk, int(m.group(1)))
    n_blocks = max_blk + 1
    print(f"Tensores mapeados: {len(gguf_names)} | blocos fisicos: {n_blocks}")
    assert n_blocks == 28, f"esperado 28 blocos, achado {n_blocks}"

    print(f"Lendo referencia: {args.ref_gguf}")
    reader = GGUFReader(args.ref_gguf, mode="r")

    out = Path(args.out)
    if out.exists():
        print(f"[REPLACE] {out}")
        out.unlink()
    writer = GGUFWriter(str(out), arch="qwen3")
    copy_kv(reader, writer, "qwen3", n_blocks)

    for st_name, gname in sorted(gguf_names.items(), key=lambda kv: kv[1]):
        t = st[st_name]
        if "norm" in gname.lower():
            data = t.to(torch.float32).cpu().numpy()
            dtype = GGMLQuantizationType.F32
        else:
            data = t.to(torch.float16).cpu().numpy()
            dtype = GGMLQuantizationType.F16
        writer.add_tensor(gname, data, raw_dtype=dtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"F16 base OK em {time.time()-t0:.1f}s: {out} ({out.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
