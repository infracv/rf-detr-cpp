#!/usr/bin/env python3
"""Sanity-check a built TensorRT engine: print bindings and cross-check the meta sidecar.

Mirrors what the C++ `rfdetr_inspect` tool does, but uses TRT's Python bindings
so it can be run before the C++ side is even built.

Example:
    python trt-files/scripts/verify_engine.py --engine trt-files/onnx/rf-detr-small.engine
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--meta", default=None,
                    help="meta sidecar path (default: <engine>.json next to the engine)")
    args = ap.parse_args()

    try:
        import tensorrt as trt
    except ImportError:
        sys.exit("tensorrt python package not installed")

    engine_path = Path(args.engine)
    if not engine_path.is_file():
        sys.exit(f"engine not found: {engine_path}")

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        sys.exit("failed to deserialize engine")

    print(f"engine: {engine_path}")
    print(f"  trt version: {trt.__version__}")

    bindings = []
    n = engine.num_io_tensors
    for i in range(n):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = list(engine.get_tensor_shape(name))
        bindings.append({
            "name": name,
            "mode": "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT",
            "dtype": str(dtype).replace("DataType.", ""),
            "shape": shape,
        })
        print(f"  [{i}] {name:<12s}  {bindings[-1]['mode']:<6s}  "
              f"{bindings[-1]['dtype']:<5s}  shape={shape}")

    meta_path = Path(args.meta) if args.meta else engine_path.with_suffix(".json")
    if not meta_path.is_file():
        print(f"\n(no meta sidecar at {meta_path})")
        return
    meta = json.loads(meta_path.read_text())
    print(f"\nmeta sidecar: {meta_path}")
    print(f"  variant      = {meta.get('variant')}")
    print(f"  input_hxw    = {meta.get('input_h')}x{meta.get('input_w')}")
    print(f"  num_queries  = {meta.get('num_queries')}")
    print(f"  num_classes  = {meta.get('num_classes')}")
    print(f"  has_masks    = {meta.get('has_masks')}")
    print(f"  precision    = {meta.get('precision')}")
    print(f"  color_order  = {meta.get('color_order')}")

    inputs = [b for b in bindings if b["mode"] == "INPUT"]
    outputs = [b for b in bindings if b["mode"] == "OUTPUT"]

    if len(inputs) != 1:
        print(f"  WARNING: {len(inputs)} input bindings, expected 1")
    elif len(inputs[0]["shape"]) >= 4:
        h_idx, w_idx = -2, -1
        eh, ew = inputs[0]["shape"][h_idx], inputs[0]["shape"][w_idx]
        mh, mw = meta.get("input_h"), meta.get("input_w")
        if eh not in (-1, mh) or ew not in (-1, mw):
            print(f"  WARNING: engine input HxW={eh}x{ew} disagrees with meta {mh}x{mw}")
        else:
            print("  input HxW matches meta")

    expected_outputs = 3 if meta.get("has_masks") else 2
    if len(outputs) != expected_outputs:
        print(f"  WARNING: {len(outputs)} outputs, expected {expected_outputs} "
              f"(has_masks={meta.get('has_masks')})")
    else:
        print(f"  output count ({len(outputs)}) matches meta")


if __name__ == "__main__":
    main()
