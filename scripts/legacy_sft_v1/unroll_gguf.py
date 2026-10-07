"""
unroll_gguf.py — Physically unrolls a qwen3loop GGUF to a standard qwen3 GGUF.

Layout (LoopSplit Fase1, 28 physical blocks -> 56 logical layers):
  prefix : blk.0  .. blk.6   (executed once  -> logical 0..6)
  middle : blk.7  .. blk.20  (executed 3x    -> logical 7..20, 21..34, 35..48)
  suffix : blk.21 .. blk.27  (executed once  -> logical 49..55)

Physical-to-logical name mapping so stock llama.cpp can run the model as a
standard 'qwen3' architecture without any patch or --override-kv flag.

Usage:
  python unroll_gguf.py <input.gguf> <output_unrolled.gguf>
"""

import sys
import re
from pathlib import Path

try:
    import gguf
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
except ImportError:
    print("ERROR: gguf package not found. pip install gguf")
    sys.exit(1)


# ── Layout ────────────────────────────────────────────────────────────────────
N_PHYS = 28
N_LOGICAL = 56

def phys_to_logical_all(phys: int) -> list[int]:
    """Return all logical layer indices for a physical block."""
    if phys <= 6:       # prefix: 0..6 (7 layers)
        return [phys]
    if phys <= 20:      # middle: 7..20 (14 layers executed 3 times)
        mid = phys - 7  # 0..13
        return [7 + mid, 21 + mid, 35 + mid]
    # suffix: 21..27 (7 layers) -> 49..55
    return [49 + (phys - 21)]


def unroll(src: Path, dst: Path):
    print(f"Reading {src} ...")
    reader = GGUFReader(str(src), mode="r")

    # ── Collect metadata ──────────────────────────────────────────────────────
    metadata = {}
    for field in reader.fields.values():
        metadata[field.name] = field

    # ── Detect architecture ───────────────────────────────────────────────────
    arch_field = metadata.get("general.architecture")
    if arch_field is None:
        print("ERROR: cannot find general.architecture in GGUF.")
        sys.exit(1)
    arch_parts = arch_field.parts
    src_arch = "".join(chr(p) for p in arch_parts[-1]).strip("\x00")
    print(f"Source Architecture : {src_arch}")

    # For stock llama.cpp compatibility, unrolled model uses standard 'qwen3'
    target_arch = "qwen3"
    print(f"Target Architecture : {target_arch}")

    # ── Count physical blocks ─────────────────────────────────────────────────
    n_layer_field = metadata.get(f"{src_arch}.block_count")
    if n_layer_field is None:
        print(f"ERROR: cannot find {src_arch}.block_count")
        sys.exit(1)
    n_phys_actual = int(n_layer_field.parts[-1][0])
    print(f"Physical blocks in source: {n_phys_actual}")

    # Total logical layers
    print(f"Logical layers (unrolled): {N_LOGICAL}")

    # ── Build writer ──────────────────────────────────────────────────────────
    print(f"\nInitializing writer for {dst} ...")
    writer = GGUFWriter(str(dst), arch=target_arch)

    # Keys to skip from original file
    skip_keys = {
        f"{src_arch}.block_count",
        f"{src_arch}.layer_map",
        "general.architecture",
        "GGUF.version",
        "GGUF.tensor_count",
        "GGUF.kv_count",
        "qwen35.triple_loop",
    }

    for name, field in metadata.items():
        if name in skip_keys:
            continue

        # Remap architecture namespace prefix if present
        if name.startswith(f"{src_arch}."):
            target_name = f"{target_arch}." + name[len(src_arch) + 1:]
        else:
            target_name = name

        try:
            ftype = field.types[0]
            val = field.parts[-1]

            if ftype == GGUFValueType.STRING:
                s = bytes(val).decode("utf-8").rstrip("\x00")
                writer.add_string(target_name, s)
            elif ftype == GGUFValueType.UINT32:
                writer.add_uint32(target_name, int(val[0]))
            elif ftype == GGUFValueType.INT32:
                writer.add_int32(target_name, int(val[0]))
            elif ftype == GGUFValueType.FLOAT32:
                writer.add_float32(target_name, float(val[0]))
            elif ftype == GGUFValueType.UINT64:
                writer.add_uint64(target_name, int(val[0]))
            elif ftype == GGUFValueType.INT64:
                writer.add_int64(target_name, int(val[0]))
            elif ftype == GGUFValueType.BOOL:
                writer.add_bool(target_name, bool(val[0]))
            elif ftype == GGUFValueType.FLOAT64:
                writer.add_float64(target_name, float(val[0]))
            elif ftype == GGUFValueType.UINT8:
                writer.add_uint8(target_name, int(val[0]))
            elif ftype == GGUFValueType.INT8:
                writer.add_int8(target_name, int(val[0]))
            elif ftype == GGUFValueType.UINT16:
                writer.add_uint16(target_name, int(val[0]))
            elif ftype == GGUFValueType.INT16:
                writer.add_int16(target_name, int(val[0]))
            elif ftype == GGUFValueType.ARRAY:
                elem_type = field.types[1]
                if elem_type == GGUFValueType.STRING:
                    # In GGUFReader, part 5 is len of elem 0, part 6 is bytes of elem 0, etc.
                    raw_strs = [bytes(p) for p in field.parts[6::2]]
                    writer.add_array(target_name, raw_strs)
                    print(f"  [ARRAY STRING] {target_name}: copied {len(raw_strs)} items")
                elif elem_type == GGUFValueType.INT32:
                    raw_ints = [int(p[0]) for p in field.parts[5:]]
                    writer.add_array(target_name, raw_ints)
                    print(f"  [ARRAY INT32]  {target_name}: copied {len(raw_ints)} items")
                elif elem_type == GGUFValueType.UINT32:
                    raw_uints = [int(p[0]) for p in field.parts[5:]]
                    writer.add_array(target_name, raw_uints)
                    print(f"  [ARRAY UINT32] {target_name}: copied {len(raw_uints)} items")
                elif elem_type == GGUFValueType.FLOAT32:
                    raw_floats = [float(p[0]) for p in field.parts[5:]]
                    writer.add_array(target_name, raw_floats)
                    print(f"  [ARRAY FLOAT32]{target_name}: copied {len(raw_floats)} items")
                else:
                    print(f"  [SKIP array] {target_name} (unsupported elem type {elem_type})")
            else:
                print(f"  [SKIP] {target_name} (unsupported type {ftype})")
        except Exception as e:
            print(f"  [WARN] could not copy field {name} -> {target_name}: {e}")

    # Write patched block_count for unrolled model
    writer.add_uint32(f"{target_arch}.block_count", N_LOGICAL)
    print(f"  [SET] {target_arch}.block_count = {N_LOGICAL}")

    # ── Map tensors ────────────────────────────────────────────────────────────
    blk_re = re.compile(r"^blk\.(\d+)\.(.+)$")
    non_blk = []
    blk_tensors: dict[int, list] = {}

    for tensor in reader.tensors:
        m = blk_re.match(tensor.name)
        if m:
            phys = int(m.group(1))
            blk_tensors.setdefault(phys, []).append(tensor)
        else:
            non_blk.append(tensor)

    print(f"\nNon-block tensors : {len(non_blk)}")
    print(f"Block tensors     : {sum(len(v) for v in blk_tensors.values())} across {len(blk_tensors)} physical blocks")

    # Add non-block tensors verbatim
    for t in non_blk:
        writer.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    # Add block tensors, expanding middle blocks
    written = 0
    for phys in sorted(blk_tensors.keys()):
        logicals = phys_to_logical_all(phys)
        for t in blk_tensors[phys]:
            m = blk_re.match(t.name)
            suffix_name = m.group(2)
            for log in logicals:
                new_name = f"blk.{log}.{suffix_name}"
                writer.add_tensor(new_name, t.data, raw_dtype=t.tensor_type)
                written += 1

    print(f"Total tensors written: {written + len(non_blk)} (expected: {len(non_blk) + N_LOGICAL * 11})")

    print("\nWriting header and metadata...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    print("Writing tensor data to disk...")
    writer.write_tensors_to_file()
    writer.close()
    print(f"\nDone! Unrolled GGUF saved successfully to: {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python unroll_gguf.py <input.gguf> <output_unrolled.gguf>")
        sys.exit(1)
    unroll(Path(sys.argv[1]), Path(sys.argv[2]))
