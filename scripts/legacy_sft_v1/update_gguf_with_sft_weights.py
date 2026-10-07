"""
update_gguf_with_sft_weights.py

Injeta os novos pesos treinados no SFT (checkpoints/qwen3loop_sft_final/model.safetensors)
no arquivo GGUF F16 (modelo_qwen3loop_sft_f16.gguf), preservando todos os metadados,
vocabulário e geometria arquitetural de 28 blocos físicos.
"""

import os
import re
import sys
from pathlib import Path
import numpy as np
import safetensors.torch

try:
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
except ImportError:
    print("ERRO: Pacote gguf não encontrado.")
    sys.exit(1)

WORK_DIR = Path(r"C:\Users\Nyx\Downloads\Diversos\qwen")
ST_PATH = WORK_DIR / "checkpoints" / "qwen3loop_sft_final" / "model.safetensors"
SRC_GGUF = WORK_DIR / "modelo_qwen3loop_sft_f16.gguf"
DST_GGUF = WORK_DIR / "modelo_qwen3loop_sft_f16.gguf.new"

def gguf_to_st_name(name: str) -> str:
    if name == "token_embd.weight":
        return "model.embed_tokens.weight"
    if name == "output_norm.weight":
        return "model.norm.weight"
    if name == "output.weight":
        return "lm_head.weight"
    m = re.match(r"^blk\.(\d+)\.(.+)$", name)
    if not m:
        return None
    i, rest = m.group(1), m.group(2)
    mapping = {
        "attn_norm.weight": f"model.layers.{i}.input_layernorm.weight",
        "ffn_norm.weight": f"model.layers.{i}.post_attention_layernorm.weight",
        "attn_q.weight": f"model.layers.{i}.self_attn.q_proj.weight",
        "attn_k.weight": f"model.layers.{i}.self_attn.k_proj.weight",
        "attn_v.weight": f"model.layers.{i}.self_attn.v_proj.weight",
        "attn_output.weight": f"model.layers.{i}.self_attn.o_proj.weight",
        "attn_q_norm.weight": f"model.layers.{i}.self_attn.q_norm.weight",
        "attn_k_norm.weight": f"model.layers.{i}.self_attn.k_norm.weight",
        "ffn_gate.weight": f"model.layers.{i}.mlp.gate_proj.weight",
        "ffn_up.weight": f"model.layers.{i}.mlp.up_proj.weight",
        "ffn_down.weight": f"model.layers.{i}.mlp.down_proj.weight",
    }
    return mapping.get(rest)

def update_gguf():
    print(f"Carregando pesos Safetensors de: {ST_PATH}")
    st = safetensors.torch.load_file(str(ST_PATH))
    print(f"Tensores carregados: {len(st)}")

    print(f"Lendo GGUF modelo base: {SRC_GGUF}")
    reader = GGUFReader(str(SRC_GGUF), mode="r")

    arch_field = reader.fields.get("general.architecture")
    arch = "".join(chr(p) for p in arch_field.parts[-1]).strip("\x00")
    print(f"Arquitetura: {arch} | Tensores GGUF: {len(reader.tensors)}")

    writer = GGUFWriter(str(DST_GGUF), arch=arch)

    # Copiar todos os metadados e vocabulários
    skip_keys = {
        "general.architecture",
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
    }

    print("Copiando campos de metadados e vocabulário...")
    for name, field in reader.fields.items():
        if name in skip_keys:
            continue
        try:
            ftype = field.types[0]
            val = field.parts[-1]

            if ftype == GGUFValueType.STRING:
                s = bytes(val).decode("utf-8").rstrip("\x00")
                writer.add_string(name, s)
            elif ftype == GGUFValueType.UINT32:
                writer.add_uint32(name, int(val[0]))
            elif ftype == GGUFValueType.INT32:
                writer.add_int32(name, int(val[0]))
            elif ftype == GGUFValueType.FLOAT32:
                writer.add_float32(name, float(val[0]))
            elif ftype == GGUFValueType.UINT64:
                writer.add_uint64(name, int(val[0]))
            elif ftype == GGUFValueType.INT64:
                writer.add_int64(name, int(val[0]))
            elif ftype == GGUFValueType.BOOL:
                writer.add_bool(name, bool(val[0]))
            elif ftype == GGUFValueType.FLOAT64:
                writer.add_float64(name, float(val[0]))
            elif ftype == GGUFValueType.UINT8:
                writer.add_uint8(name, int(val[0]))
            elif ftype == GGUFValueType.INT8:
                writer.add_int8(name, int(val[0]))
            elif ftype == GGUFValueType.UINT16:
                writer.add_uint16(name, int(val[0]))
            elif ftype == GGUFValueType.INT16:
                writer.add_int16(name, int(val[0]))
            elif ftype == GGUFValueType.ARRAY:
                elem_type = field.types[1]
                if elem_type == GGUFValueType.STRING:
                    raw_strs = [bytes(p) for p in field.parts[6::2]]
                    writer.add_array(name, raw_strs)
                elif elem_type == GGUFValueType.INT32:
                    raw_ints = [int(p[0]) for p in field.parts[5:]]
                    writer.add_array(name, raw_ints)
                elif elem_type == GGUFValueType.UINT32:
                    raw_uints = [int(p[0]) for p in field.parts[5:]]
                    writer.add_array(name, raw_uints)
                elif elem_type == GGUFValueType.FLOAT32:
                    raw_floats = [float(p[0]) for p in field.parts[5:]]
                    writer.add_array(name, raw_floats)
        except Exception as e:
            print(f"  [AVISO] Não foi possível copiar campo {name}: {e}")

    # Atualizar tensores
    print("\nInjetando tensores com pesos SFT atualizados...")
    updated_count = 0
    for i, tensor in enumerate(reader.tensors):
        st_name = gguf_to_st_name(tensor.name)
        if st_name is None or st_name not in st:
            raise KeyError(f"Tensor {tensor.name} não encontrado no Safetensors! ({st_name})")

        st_tensor = st[st_name]

        # Garantir tipo de dado compatível
        if tensor.tensor_type == GGMLQuantizationType.F16:
            data = st_tensor.to(safetensors.torch.torch.float16).cpu().numpy()
        elif tensor.tensor_type == GGMLQuantizationType.F32:
            data = st_tensor.to(safetensors.torch.torch.float32).cpu().numpy()
        else:
            data = st_tensor.cpu().numpy()

        writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
        updated_count += 1
        if (i + 1) % 50 == 0 or (i + 1) == len(reader.tensors):
            print(f"  [{i+1}/{len(reader.tensors)}] Injetado: {tensor.name} ({data.dtype})")

    print(f"\nTotal de tensores injetados: {updated_count}")
    print("Gravando cabeçalhos e metadados...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    print("Gravando dados dos tensores...")
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nArquivo temporário gerado: {DST_GGUF} ({DST_GGUF.stat().st_size/1e9:.2f} GB)")

    # Substituir arquivo original com segurança
    backup_gguf = WORK_DIR / "modelo_qwen3loop_sft_f16.gguf.old"
    if backup_gguf.exists():
        backup_gguf.unlink()
    SRC_GGUF.rename(backup_gguf)
    DST_GGUF.rename(SRC_GGUF)
    print(f"Sucesso! {SRC_GGUF.name} foi atualizado com os novos pesos do SFT!")

if __name__ == "__main__":
    update_gguf()
