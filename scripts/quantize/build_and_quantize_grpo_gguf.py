"""
build_and_quantize_grpo_gguf.py
Pipeline completo automatizado:
1. Merge dos adaptadores LoRA (checkpoints/qwen3loop_tool_grpo_final) no modelo base SFT (checkpoints/qwen3loop_sft_final)
2. Injeção dos novos pesos Safetensors no GGUF F16 (models/gguf/modelo_qwen3loop_sft_f16.gguf)
3. Desenrolamento físico (unroll) de 28 blocos para 56 camadas lógicas compatíveis com llama.cpp (arch=qwen3)
4. Quantização F16 -> Q8_0 (models/gguf/unrolled_modelo_qwen3loop_grpo_q8_0.gguf)
"""

import os
import sys
import re
import time
import shutil
from pathlib import Path
import numpy as np
import torch

# Adicionar caminhos para engine
sys.path.insert(0, os.path.abspath("engine"))
sys.path.insert(0, os.path.abspath("."))

from transformers import AutoModelForCausalLM, AutoConfig
from peft import PeftModel
import safetensors.torch
import qwen3loop_python  # noqa: F401

try:
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
except ImportError:
    print("ERRO: Pacote gguf não encontrado.")
    sys.exit(1)

ROOT_DIR = Path(os.path.abspath("."))
SFT_DIR = ROOT_DIR / "checkpoints" / "qwen3loop_sft_final"
LORA_DIR = ROOT_DIR / "checkpoints" / "qwen3loop_tool_grpo_final"
MERGED_DIR = ROOT_DIR / "checkpoints" / "qwen3loop_tool_grpo_merged"

TEMPLATE_F16_GGUF = ROOT_DIR / "models" / "gguf" / "modelo_qwen3loop_sft_f16.gguf"
MERGED_F16_GGUF = ROOT_DIR / "models" / "gguf" / "modelo_qwen3loop_grpo_f16.gguf"
UNROLLED_F16_GGUF = ROOT_DIR / "models" / "gguf" / "unrolled_modelo_qwen3loop_grpo_f16.gguf"
FINAL_Q8_GGUF = ROOT_DIR / "models" / "gguf" / "unrolled_modelo_qwen3loop_grpo_q8_0.gguf"


# -----------------------------------------------------------------------------
# STEP 1: Fusão do LoRA no Modelo Base
# -----------------------------------------------------------------------------
def step1_merge_lora():
    print("\n" + "=" * 70)
    print("STEP 1: FUSÃO DOS PESOS LORA NO MODELO BASE SFT")
    print("=" * 70)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    merged_st = MERGED_DIR / "model.safetensors"

    if merged_st.exists():
        print(f"[SKIP] Pesos já fundidos encontrados em: {merged_st}")
        return merged_st

    t0 = time.time()
    print(f"Carregando modelo base SFT de: {SFT_DIR}")
    config = AutoConfig.from_pretrained(str(SFT_DIR))
    base_model = AutoModelForCausalLM.from_pretrained(
        str(SFT_DIR),
        config=config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    print(f"Acoplando adaptadores LoRA de: {LORA_DIR}")
    lora_model = PeftModel.from_pretrained(base_model, str(LORA_DIR))

    print("Executando merge_and_unload()...")
    merged = lora_model.merge_and_unload()

    print(f"Gravando modelo fundido em: {MERGED_DIR} ...")
    merged.save_pretrained(str(MERGED_DIR))
    print(f"Merge concluído com sucesso em {time.time() - t0:.2f}s!")
    return merged_st


# -----------------------------------------------------------------------------
# STEP 2: Injeção dos Pesos Fundidos no GGUF F16
# -----------------------------------------------------------------------------
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


def step2_inject_weights(st_path: Path):
    print("\n" + "=" * 70)
    print("STEP 2: INJEÇÃO DOS PESOS NO GGUF F16")
    print("=" * 70)

    if MERGED_F16_GGUF.exists():
        print(f"[SKIP] GGUF F16 fundido já existe: {MERGED_F16_GGUF}")
        return

    t0 = time.time()
    print(f"Lendo pesos Safetensors de: {st_path}")
    st = safetensors.torch.load_file(str(st_path))

    print(f"Lendo template GGUF: {TEMPLATE_F16_GGUF}")
    reader = GGUFReader(str(TEMPLATE_F16_GGUF), mode="r")

    arch_field = reader.fields.get("general.architecture")
    arch = "".join(chr(p) for p in arch_field.parts[-1]).strip("\x00")
    print(f"Arquitetura: {arch} | Tensores: {len(reader.tensors)}")

    writer = GGUFWriter(str(MERGED_F16_GGUF), arch=arch)

    skip_keys = {
        "general.architecture",
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
    }

    for name, field in reader.fields.items():
        if name in skip_keys:
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
                elem_type = field.types[1]
                if elem_type == GGUFValueType.STRING:
                    writer.add_array(name, [bytes(p) for p in field.parts[6::2]])
                elif elem_type == GGUFValueType.INT32:
                    writer.add_array(name, [int(p[0]) for p in field.parts[5:]])
                elif elem_type == GGUFValueType.UINT32:
                    writer.add_array(name, [int(p[0]) for p in field.parts[5:]])
                elif elem_type == GGUFValueType.FLOAT32:
                    writer.add_array(name, [float(p[0]) for p in field.parts[5:]])
        except Exception:
            pass

    for tensor in reader.tensors:
        st_name = gguf_to_st_name(tensor.name)
        if st_name is None or st_name not in st:
            raise KeyError(f"Tensor {tensor.name} não encontrado no Safetensors! ({st_name})")

        st_tensor = st[st_name]
        if tensor.tensor_type == GGMLQuantizationType.F16:
            data = st_tensor.to(torch.float16).cpu().numpy()
        elif tensor.tensor_type == GGMLQuantizationType.F32:
            data = st_tensor.to(torch.float32).cpu().numpy()
        else:
            data = st_tensor.cpu().numpy()

        writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"GGUF F16 gerado em {time.time() - t0:.2f}s: {MERGED_F16_GGUF} ({MERGED_F16_GGUF.stat().st_size/1e9:.2f} GB)")


# -----------------------------------------------------------------------------
# STEP 3: Desenrolamento Físico (28 blocos -> 56 camadas lógicas qwen3)
# -----------------------------------------------------------------------------
def phys_to_logical_all(phys: int) -> list:
    if phys <= 6:       # prefix: 0..6 (7 layers)
        return [phys]
    if phys <= 20:      # middle: 7..20 (14 layers executed 3 times)
        mid = phys - 7
        return [7 + mid, 21 + mid, 35 + mid]
    # suffix: 21..27 (7 layers) -> 49..55
    return [49 + (phys - 21)]


def step3_unroll_gguf():
    print("\n" + "=" * 70)
    print("STEP 3: DESENROLAMENTO FÍSICO (UNROLL 28 -> 56 CAMADAS)")
    print("=" * 70)

    if UNROLLED_F16_GGUF.exists():
        print(f"[SKIP] GGUF unrolled F16 já existe: {UNROLLED_F16_GGUF}")
        return

    t0 = time.time()
    reader = GGUFReader(str(MERGED_F16_GGUF), mode="r")
    target_arch = "qwen3"
    writer = GGUFWriter(str(UNROLLED_F16_GGUF), arch=target_arch)

    skip_keys = {
        "qwen3loop.block_count",
        "qwen3loop.layer_map",
        "general.architecture",
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "qwen35.triple_loop",
    }

    for name, field in reader.fields.items():
        if name in skip_keys: continue
        clean_name = name.replace("qwen3loop.", f"{target_arch}.")
        try:
            ftype = field.types[0]
            val = field.parts[-1]
            if ftype == GGUFValueType.STRING:
                writer.add_string(clean_name, bytes(val).decode("utf-8").rstrip("\x00"))
            elif ftype == GGUFValueType.UINT32: writer.add_uint32(clean_name, int(val[0]))
            elif ftype == GGUFValueType.INT32: writer.add_int32(clean_name, int(val[0]))
            elif ftype == GGUFValueType.FLOAT32: writer.add_float32(clean_name, float(val[0]))
            elif ftype == GGUFValueType.UINT64: writer.add_uint64(clean_name, int(val[0]))
            elif ftype == GGUFValueType.INT64: writer.add_int64(clean_name, int(val[0]))
            elif ftype == GGUFValueType.BOOL: writer.add_bool(clean_name, bool(val[0]))
            elif ftype == GGUFValueType.FLOAT64: writer.add_float64(clean_name, float(val[0]))
            elif ftype == GGUFValueType.UINT8: writer.add_uint8(clean_name, int(val[0]))
            elif ftype == GGUFValueType.INT8: writer.add_int8(clean_name, int(val[0]))
            elif ftype == GGUFValueType.UINT16: writer.add_uint16(clean_name, int(val[0]))
            elif ftype == GGUFValueType.INT16: writer.add_int16(clean_name, int(val[0]))
            elif ftype == GGUFValueType.ARRAY:
                elem_type = field.types[1]
                if elem_type == GGUFValueType.STRING:
                    writer.add_array(clean_name, [bytes(p) for p in field.parts[6::2]])
                elif elem_type == GGUFValueType.INT32:
                    writer.add_array(clean_name, [int(p[0]) for p in field.parts[5:]])
                elif elem_type == GGUFValueType.UINT32:
                    writer.add_array(clean_name, [int(p[0]) for p in field.parts[5:]])
                elif elem_type == GGUFValueType.FLOAT32:
                    writer.add_array(clean_name, [float(p[0]) for p in field.parts[5:]])
        except Exception:
            pass

    writer.add_uint32(f"{target_arch}.block_count", 56)

    # Duplicação e mapeamento dos tensores
    blk_re = re.compile(r"^blk\.(\d+)\.(.+)$")
    for tensor in reader.tensors:
        tname = tensor.name
        m = blk_re.match(tname)
        if m:
            phys_idx = int(m.group(1))
            suffix = m.group(2)
            logical_indices = phys_to_logical_all(phys_idx)
            for log_idx in logical_indices:
                new_tname = f"blk.{log_idx}.{suffix}"
                writer.add_tensor(new_tname, tensor.data, raw_dtype=tensor.tensor_type)
        else:
            writer.add_tensor(tname, tensor.data, raw_dtype=tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"GGUF Unrolled F16 gerado em {time.time() - t0:.2f}s: {UNROLLED_F16_GGUF} ({UNROLLED_F16_GGUF.stat().st_size/1e9:.2f} GB)")


# -----------------------------------------------------------------------------
# STEP 4: Quantização F16 -> Q8_0
# -----------------------------------------------------------------------------
def quantize_to_q8_0(data: np.ndarray) -> np.ndarray:
    """Quantiza float32/float16 para Q8_0 em blocos de 32 elementos."""
    orig_shape = data.shape
    flat = data.astype(np.float32).flatten()
    n_elements = flat.shape[0]
    block_size = 32
    assert n_elements % block_size == 0, f"n_elements {n_elements} not divisible by 32"
    n_blocks = n_elements // block_size

    blocks = flat.reshape(n_blocks, block_size)
    d = np.max(np.abs(blocks), axis=1) / 127.0
    d_f16 = d.astype(np.float16)

    d_safe = np.where(d == 0, 1.0, d)[:, np.newaxis]
    qs = np.clip(np.round(blocks / d_safe), -128, 127).astype(np.int8)

    out = bytearray(n_blocks * 34)
    out_mv = memoryview(out)

    d_bytes = d_f16.tobytes()
    qs_bytes = qs.tobytes()

    for b in range(n_blocks):
        offset = b * 34
        out_mv[offset:offset+2] = d_bytes[b*2:(b+1)*2]
        out_mv[offset+2:offset+34] = qs_bytes[b*32:(b+1)*32]

    return np.frombuffer(out, dtype=np.uint8).reshape((*orig_shape[:-1], orig_shape[-1] // 32 * 34))


def step4_quantize_q8():
    print("\n" + "=" * 70)
    print("STEP 4: QUANTIZAÇÃO F16 -> Q8_0")
    print("=" * 70)

    if FINAL_Q8_GGUF.exists():
        print(f"[SKIP] Modelo Q8 final já existe: {FINAL_Q8_GGUF}")
        return

    t0 = time.time()
    reader = GGUFReader(str(UNROLLED_F16_GGUF), mode="r")
    target_arch = "qwen3"
    writer = GGUFWriter(str(FINAL_Q8_GGUF), arch=target_arch)

    skip_keys = {
        "general.architecture",
        "general.file_type",
        "general.quantization_version",
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
    }

    for name, field in reader.fields.items():
        if name in skip_keys: continue
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
                elem_type = field.types[1]
                if elem_type == GGUFValueType.STRING:
                    writer.add_array(name, [bytes(p) for p in field.parts[6::2]])
                elif elem_type == GGUFValueType.INT32:
                    writer.add_array(name, [int(p[0]) for p in field.parts[5:]])
                elif elem_type == GGUFValueType.UINT32:
                    writer.add_array(name, [int(p[0]) for p in field.parts[5:]])
                elif elem_type == GGUFValueType.FLOAT32:
                    writer.add_array(name, [float(p[0]) for p in field.parts[5:]])
        except Exception:
            pass

    writer.add_uint32("general.file_type", int(GGMLQuantizationType.Q8_0))
    writer.add_uint32("general.quantization_version", 2)

    total_t = len(reader.tensors)
    for i, tensor in enumerate(reader.tensors):
        name = tensor.name
        shape = tensor.shape
        data = tensor.data
        n_elements = data.size

        # Decisão de quantização
        is_norm = "norm" in name.lower()
        can_quant = (len(shape) > 1 and n_elements > 64 and (n_elements % 32 == 0) and not is_norm)

        if can_quant:
            q_data = quantize_to_q8_0(data)
            writer.add_tensor(name, q_data, raw_dtype=GGMLQuantizationType.Q8_0)
        else:
            if is_norm or len(shape) <= 1:
                f32_data = data.astype(np.float32)
                writer.add_tensor(name, f32_data, raw_dtype=GGMLQuantizationType.F32)
            else:
                writer.add_tensor(name, data, raw_dtype=tensor.tensor_type)

        if (i + 1) % 100 == 0 or (i + 1) == total_t:
            print(f"  [{i+1}/{total_t}] Processado: {name}")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nMODELO FINAL QUANTIZADO EM Q8_0: {FINAL_Q8_GGUF} ({FINAL_Q8_GGUF.stat().st_size/1e9:.2f} GB)")
    print(f"Tempo total de quantização: {time.time() - t0:.2f} segundos!")


def main():
    start_total = time.time()
    st_path = step1_merge_lora()
    step2_inject_weights(st_path)
    step3_unroll_gguf()
    step4_quantize_q8()
    total_time = (time.time() - start_total) / 60
    print("\n" + "=" * 70)
    print(f"PIPELINE CONCLUÍDO COM SUCESSO EM {total_time:.2f} MINUTOS!")
    print(f"Arquivo gerado para inferência ultrarrápida:")
    print(f"-> {FINAL_Q8_GGUF}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
