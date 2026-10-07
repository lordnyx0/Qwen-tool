"""
quantize_q8_0.py  --  Quantiza um GGUF F16 para Q8_0 usando gguf Python puro.
Funciona com qualquer arquitetura (nao depende do llama-quantize.exe).

Regras de quantizacao (mesmas do llama.cpp):
  - Tensores 1D (normas, biases)       -> mantidos em F32
  - token_embd.weight                  -> mantido em Q8_0 (como llama.cpp faz)
  - output.weight (lm_head)            -> Q8_0
  - Todos os outros pesos 2D+          -> Q8_0
  - Tensores muito pequenos (<= 64 el) -> F32

Usage:
  python quantize_q8_0.py <input_f16.gguf> <output_q8.gguf>
"""

import sys
import numpy as np
from pathlib import Path

try:
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGUFValueType
except ImportError:
    print("ERROR: gguf package not found. pip install gguf")
    sys.exit(1)


def should_quantize(name: str, shape: tuple) -> bool:
    """Decide se o tensor deve ser quantizado."""
    n_elements = 1
    for d in shape:
        n_elements *= d
    # Nao quantizar tensores muito pequenos ou 1D
    if len(shape) <= 1 or n_elements <= 64:
        return False
    # Nao quantizar norms
    lower = name.lower()
    if "norm" in lower:
        return False
    return True


def quantize_file(src: Path, dst: Path):
    print(f"Lendo  : {src} ({src.stat().st_size/1e9:.2f} GB)")
    reader = GGUFReader(str(src), mode="r")

    # ── Detectar arch ──────────────────────────────────────────────────────
    arch_field = reader.fields.get("general.architecture")
    arch_parts = arch_field.parts
    arch = "".join(chr(p) for p in arch_parts[-1]).strip("\x00")
    print(f"Arch   : {arch}")
    print(f"Tensors: {len(reader.tensors)}")

    writer = GGUFWriter(str(dst), arch=arch)

    # ── Copiar metadata ────────────────────────────────────────────────────
    skip_keys = {
        "general.architecture",
        "general.file_type",
        "general.quantization_version",
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
                    print(f"  [ARRAY STRING] {name}: {len(raw_strs)} items")
                elif elem_type == GGUFValueType.INT32:
                    raw_ints = [int(p[0]) for p in field.parts[5:]]
                    writer.add_array(name, raw_ints)
                    print(f"  [ARRAY INT32]  {name}: {len(raw_ints)} items")
                elif elem_type == GGUFValueType.UINT32:
                    raw_uints = [int(p[0]) for p in field.parts[5:]]
                    writer.add_array(name, raw_uints)
                    print(f"  [ARRAY UINT32] {name}: {len(raw_uints)} items")
                elif elem_type == GGUFValueType.FLOAT32:
                    raw_floats = [float(p[0]) for p in field.parts[5:]]
                    writer.add_array(name, raw_floats)
                    print(f"  [ARRAY FLOAT32]{name}: {len(raw_floats)} items")
                else:
                    print(f"  [SKIP array] {name} (unsupported elem type {elem_type})")
            else:
                print(f"  [SKIP] {name} (unsupported type {ftype})")
        except Exception as e:
            print(f"  [WARN] could not copy field {name}: {e}")

    # file_type = Q8_0 (8), quantization_version = 2
    writer.add_uint32("general.file_type", 8)
    writer.add_uint32("general.quantization_version", 2)

    # ── Quantizar tensores ─────────────────────────────────────────────────
    n_quant = 0
    n_keep = 0
    total = len(reader.tensors)
    for i, tensor in enumerate(reader.tensors):
        shape = tuple(tensor.shape)
        do_quant = should_quantize(tensor.name, shape)

        data = tensor.data
        if tensor.tensor_type == GGMLQuantizationType.F16:
            data_f32 = data.view(np.float16).astype(np.float32)
        elif tensor.tensor_type == GGMLQuantizationType.F32:
            data_f32 = data.view(np.float32)
        else:
            # Ja quantizado ou formato desconhecido -> copiar verbatim
            writer.add_tensor(tensor.name, data, raw_dtype=tensor.tensor_type)
            n_keep += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{total}] {tensor.name} (verbatim)")
            continue

        if do_quant:
            q_data = gguf.quantize(data_f32, GGMLQuantizationType.Q8_0)
            writer.add_tensor(tensor.name, q_data, raw_dtype=GGMLQuantizationType.Q8_0)
            n_quant += 1
        else:
            # Guardar em F32
            writer.add_tensor(tensor.name, data_f32, raw_dtype=GGMLQuantizationType.F32)
            n_keep += 1

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] quantizados={n_quant} mantidos_f32={n_keep}")

    print(f"\nEscrevendo {dst} ...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size_gb = dst.stat().st_size / 1e9
    print(f"Concluido: {dst} ({size_gb:.2f} GB)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python quantize_q8_0.py <input_f16.gguf> <output_q8.gguf>")
        sys.exit(1)
    quantize_file(Path(sys.argv[1]), Path(sys.argv[2]))
