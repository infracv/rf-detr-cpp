#!/usr/bin/env python3
"""slim_onnx.py — Simplify an RF-DETR ONNX graph with onnxslim.

RF-DETR's exported graph is heavy with reshape/transpose/gather bookkeeping that
TensorRT's builder does not fully fuse on its own. Running onnxslim first
constant-folds shapes and fuses redundant ops, roughly halving the node count and
letting TRT produce a faster engine.

SAFE FOR FP32 ONLY. onnxslim is numerically lossless on FP32 (verified: max abs
diff ~1e-5, identical detections), but its fusions reduce FP16 numerical headroom
and trigger RF-DETR's known attention overflow — degrading FP16 accuracy. Run this
on the FP32 ONNX, before any FP16 conversion. Do not slim an already-FP16 model.

Usage:
    python trt-files/scripts/slim_onnx.py \\
        --onnx trt-files/onnx/rf-detr-nano.onnx \\
        --out  trt-files/onnx/rf-detr-nano-slim.onnx

Then build the FP32 engine as usual:
    python trt-files/scripts/build_engine.py \\
        --onnx trt-files/onnx/rf-detr-nano-slim.onnx --precision fp32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import onnx
from onnxslim import slim


def slim_model(model: onnx.ModelProto) -> onnx.ModelProto:
    """Simplify an ONNX graph in place-ish; returns the slimmed model."""
    return slim(model)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="Input FP32 ONNX model")
    ap.add_argument("--out", default=None,
                    help="Output slimmed ONNX path (default: <stem>-slim.onnx)")
    ap.add_argument("--meta", default=None,
                    help="Input meta sidecar (default: <onnx>.json)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.is_file():
        sys.exit(f"error: ONNX not found: {onnx_path}")

    out_path = Path(args.out) if args.out else onnx_path.parent / (onnx_path.stem + "-slim.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[slim_onnx] loading  {onnx_path}")
    model = onnx.load(str(onnx_path))
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            sys.exit("error: model input is FP16 — onnxslim degrades FP16 accuracy; "
                     "slim the FP32 ONNX before converting to FP16.")
    n_before = len(model.graph.node)

    print(f"[slim_onnx] simplifying ({n_before} nodes) ...")
    model = slim_model(model)
    n_after = len(model.graph.node)

    print(f"[slim_onnx] checking model ...")
    try:
        onnx.checker.check_model(model, full_check=False)
        print("  ONNX check: OK")
    except onnx.checker.ValidationError as e:
        print(f"  ONNX check warning: {e} (may still build fine with TRT)")

    print(f"[slim_onnx] saving   {out_path}")
    onnx.save(model, str(out_path))
    print(f"[slim_onnx] nodes: {n_before} -> {n_after} "
          f"({100.0 * (n_before - n_after) / max(n_before, 1):.1f}% fewer)")

    # Copy meta sidecar so downstream build_engine.py finds it.
    meta_in = Path(args.meta) if args.meta else onnx_path.with_suffix(".json")
    if meta_in.is_file():
        meta = json.loads(meta_in.read_text())
        out_meta = out_path.with_suffix(".json")
        out_meta.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"[slim_onnx] meta     {out_meta}")

    print("[slim_onnx] done.")


if __name__ == "__main__":
    main()
