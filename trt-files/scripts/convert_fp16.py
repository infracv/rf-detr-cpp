#!/usr/bin/env python3
"""convert_fp16.py — Convert an RF-DETR FP32 ONNX model to FP16.

For TensorRT 11+ (strongly-typed networks), FP16 engines require an FP16 ONNX.
This script:
  1. Converts all FP32 initializers (weights) to FP16.
  2. Updates all internal value_info tensor types from FLOAT → FLOAT16.
  3. Converts Cast(to=FLOAT) nodes to Cast(to=FLOAT16) for internal attention casts.
  4. Keeps graph inputs and outputs as FLOAT32 so the C++ runtime
     (which always passes float* buffers) works unchanged.
  5. Keeps non-float tensors (INT64, BOOL, INT32) untouched.

For TRT < 11, use:
    rfdetr_build --onnx model.onnx --precision fp16
which sets kFP16 on the builder config directly.

Usage:
    python trt-files/scripts/convert_fp16.py \\
        --onnx trt-files/onnx/rf-detr-nano.onnx \\
        --out  trt-files/onnx/rf-detr-nano-fp16.onnx

Then build the engine (any TRT version):
    rfdetr_build --onnx trt-files/onnx/rf-detr-nano-fp16.onnx --precision fp32
    # (engine will be FP16 internally because ONNX weights are FP16)
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def float32_initializer_to_float16(init: TensorProto) -> TensorProto:
    """Convert a single FP32 initializer to FP16."""
    arr = numpy_helper.to_array(init).astype(np.float16)
    new_init = numpy_helper.from_array(arr, name=init.name)
    return new_init


def convert_to_fp16(model: onnx.ModelProto,
                    keep_io_fp32: bool = True) -> onnx.ModelProto:
    """Convert all FP32 parts of the model to FP16.

    Args:
        model:         Input FP32 ONNX model.
        keep_io_fp32:  Keep graph inputs/outputs as FLOAT32 (recommended for TRT
                       so the C++ runtime doesn't need to change input/output buffers).
    """
    graph = model.graph

    # Collect graph input and output names to optionally preserve their dtype.
    io_names: set[str] = set()
    if keep_io_fp32:
        io_names = {t.name for t in graph.input} | {t.name for t in graph.output}

    # 1. Convert FP32 initializers → FP16.
    new_initializers = []
    converted = set()
    for init in graph.initializer:
        if init.data_type == TensorProto.FLOAT and init.name not in io_names:
            new_initializers.append(float32_initializer_to_float16(init))
            converted.add(init.name)
        else:
            new_initializers.append(init)

    del graph.initializer[:]
    graph.initializer.extend(new_initializers)

    # 2. Update value_info dtype FLOAT → FLOAT16 (intermediate tensors).
    for vi in graph.value_info:
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT and vi.name not in io_names:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT16

    # 3. Update graph input types (unless we're keeping them FP32).
    if not keep_io_fp32:
        for inp in graph.input:
            if inp.type.tensor_type.elem_type == TensorProto.FLOAT:
                inp.type.tensor_type.elem_type = TensorProto.FLOAT16
        for out in graph.output:
            if out.type.tensor_type.elem_type == TensorProto.FLOAT:
                out.type.tensor_type.elem_type = TensorProto.FLOAT16

    # 4. Convert FP32 tensor-valued node attributes → FP16.
    #    Constant carries a full tensor; ConstantOfShape carries the fill value,
    #    and its dtype decides the output dtype. Missing ConstantOfShape leaves a
    #    Float tensor feeding Half arithmetic, which TRT rejects outright
    #    ("ElementWiseOperation PROD must have same input types"). Static-batch
    #    exports never hit it because the subgraph constant-folds away; with a
    #    dynamic batch axis it survives to the parser.
    const_converted = 0
    output_names_set = {t.name for t in graph.output}
    for node in graph.node:
        if node.op_type not in ("Constant", "ConstantOfShape"):
            continue
        # Skip if this constant feeds a graph output directly.
        if any(o in output_names_set for o in node.output):
            continue
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT:
                arr = numpy_helper.to_array(attr.t).astype(np.float16)
                new_t = numpy_helper.from_array(arr)
                attr.t.CopyFrom(new_t)
                const_converted += 1
    print(f"  Constant/ConstantOfShape(FLOAT) : {const_converted}")

    # 5. Patch Cast(to=FLOAT) → Cast(to=FLOAT16) for internal casts.
    #    These are attention-internal casts that must match the new dtype.
    #    Do NOT patch casts that feed graph outputs or casts that produce INT64/BOOL.
    patched = 0
    output_names = {t.name for t in graph.output}
    for node in graph.node:
        if node.op_type != "Cast":
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.FLOAT:
                # Skip if any output feeds the graph outputs (keep those FP32).
                if any(o in output_names for o in node.output):
                    continue
                attr.i = TensorProto.FLOAT16
                patched += 1
                break

    # 6. Re-insert an input cast when a kept-FP32 input feeds FP16 compute with no
    #    cast in between. In the FP32 graph, input→Cast(f32→f32)→conv is a no-op,
    #    so graph simplifiers (onnxslim) legitimately delete it — leaving an FP32
    #    input wired straight into an FP16 kernel, which TRT rejects
    #    ("input and kernel must be of same type"). Bridge it back.
    inserted = 0
    if keep_io_fp32:
        # Metadata ops are dtype-agnostic; leaving them on the FP32 input is fine.
        DTYPE_AGNOSTIC = {"Shape", "Size"}
        producers = {o for n in graph.node for o in n.output}
        for inp in graph.input:
            if inp.type.tensor_type.elem_type != TensorProto.FLOAT:
                continue
            consumers = [n for n in graph.node if inp.name in n.input]
            # Already bridged? (normal, non-simplified export)
            already = any(
                n.op_type == "Cast" and any(
                    a.name == "to" and a.i == TensorProto.FLOAT16 for a in n.attribute)
                for n in consumers)
            needs = [n for n in consumers if n.op_type not in DTYPE_AGNOSTIC
                     and n.op_type != "Cast"]
            if already or not needs:
                continue
            cast_out = inp.name + "_fp16"
            while cast_out in producers:
                cast_out += "_"
            cast_node = helper.make_node(
                "Cast", inputs=[inp.name], outputs=[cast_out],
                to=TensorProto.FLOAT16, name=inp.name + "/InputCastFP16")
            graph.node.insert(0, cast_node)
            for n in needs:
                for i, name in enumerate(n.input):
                    if name == inp.name:
                        n.input[i] = cast_out
            inserted += 1

    print(f"  Initializers converted to FP16 : {len(converted)}")
    print(f"  Cast(to=FLOAT) → Cast(to=FP16): {patched}")
    print(f"  Input casts re-inserted        : {inserted}")

    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx",  required=True, help="Input FP32 ONNX model")
    ap.add_argument("--out",   default=None,  help="Output FP16 ONNX path (default: <stem>-fp16.onnx)")
    ap.add_argument("--full-fp16", action="store_true",
                    help="Also convert graph I/O to FP16 (not recommended for TRT)")
    ap.add_argument("--meta",  default=None,
                    help="Input meta sidecar (default: <onnx>.json)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.is_file():
        sys.exit(f"error: ONNX not found: {onnx_path}")

    out_path = Path(args.out) if args.out else onnx_path.parent / (onnx_path.stem + "-fp16.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[convert_fp16] loading  {onnx_path}")
    model = onnx.load(str(onnx_path))

    print(f"[convert_fp16] converting to FP16 (keep_io_fp32={not args.full_fp16}) ...")
    model = convert_to_fp16(model, keep_io_fp32=not args.full_fp16)

    print(f"[convert_fp16] checking model ...")
    try:
        onnx.checker.check_model(model, full_check=False)
        print("  ONNX check: OK")
    except onnx.checker.ValidationError as e:
        print(f"  ONNX check warning: {e} (may still build fine with TRT)")

    print(f"[convert_fp16] saving   {out_path}")
    onnx.save(model, str(out_path))

    # Copy and update meta sidecar.
    meta_in = Path(args.meta) if args.meta else onnx_path.with_suffix(".json")
    if meta_in.is_file():
        meta = json.loads(meta_in.read_text())
        meta["precision"] = "fp16"
        out_meta = out_path.with_suffix(".json")
        out_meta.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[convert_fp16] meta     {out_meta}")

    print("[convert_fp16] done.")
    print()
    print("Next step — build TRT engine:")
    print(f"  rfdetr_build --onnx {out_path} --precision fp32")
    print(f"  # engine will run FP16 internally (weights are FP16)")


if __name__ == "__main__":
    main()
