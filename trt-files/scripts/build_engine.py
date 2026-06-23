#!/usr/bin/env python3
"""Build a TensorRT .engine from an ONNX model via `trtexec`.

Reads the ONNX's meta sidecar (produced by `export_onnx.py`), invokes
`trtexec` with the right shape/precision flags, and writes an updated
sidecar next to the engine.

Example:
    python trt-files/scripts/build_engine.py \\
        --onnx trt-files/onnx/rf-detr-small.onnx \\
        --precision fp16

INT8 (calibration cache must already exist; see int8_calibrate.py in step 9):
    python trt-files/scripts/build_engine.py \\
        --onnx trt-files/onnx/rf-detr-small.onnx \\
        --precision int8 \\
        --int8-calib trt-files/onnx/rf-detr-small.calib
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--meta", default=None,
                    help="input meta sidecar (default: <onnx>.json next to the ONNX)")
    ap.add_argument("--out", default=None,
                    help="output engine path (default: <onnx>.engine)")
    ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
    ap.add_argument("--int8-calib", default=None, help="(int8 only) path to .calib cache")
    ap.add_argument("--workspace-mib", type=int, default=4096,
                    help="workspace memory pool size in MiB (default: 4096)")
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--opt-batch", type=int, default=1)
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--input-name", default="input",
                    help="input tensor name in the ONNX graph (default: input)")
    ap.add_argument("--trtexec", default="trtexec",
                    help="path to trtexec binary (default: PATH lookup)")
    ap.add_argument("--extra", action="append", default=[],
                    help="extra trtexec arg, may be repeated; e.g. --extra=--useCudaGraph")
    args = ap.parse_args()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.is_file():
        sys.exit(f"onnx not found: {onnx_path}")

    in_meta_path = Path(args.meta) if args.meta else onnx_path.with_suffix(".json")
    if not in_meta_path.is_file():
        sys.exit(f"meta sidecar not found: {in_meta_path}\n"
                 f"  run export_onnx.py first or pass --meta")
    meta = json.loads(in_meta_path.read_text())

    engine_path = Path(args.out) if args.out else onnx_path.with_suffix(".engine")
    out_meta = engine_path.with_suffix(".json")

    if shutil.which(args.trtexec) is None and not Path(args.trtexec).is_file():
        sys.exit(f"trtexec not found: {args.trtexec}")

    H, W = meta["input_h"], meta["input_w"]
    cmd = [
        args.trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{args.workspace_mib}",
    ]

    if args.precision == "fp16":
        cmd.append("--fp16")
    elif args.precision == "int8":
        cmd.append("--int8")
        # FP16 fallback for layers that can't be quantized cleanly.
        cmd.append("--fp16")
        if args.int8_calib:
            calib = Path(args.int8_calib)
            if not calib.is_file():
                sys.exit(f"int8 calibration cache not found: {calib}")
            cmd.append(f"--calib={calib}")

    dyn = bool(meta.get("dynamic_batch"))
    if dyn or args.max_batch > 1:
        n = args.input_name
        cmd += [
            f"--minShapes={n}:{args.min_batch}x3x{H}x{W}",
            f"--optShapes={n}:{args.opt_batch}x3x{H}x{W}",
            f"--maxShapes={n}:{args.max_batch}x3x{H}x{W}",
        ]

    cmd += list(args.extra)

    print("[build_engine] running:")
    print("  " + " ".join(map(str, cmd)))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"trtexec failed with exit code {rc}")

    meta["precision"] = args.precision
    meta["dynamic_batch"] = dyn or args.max_batch > 1
    meta["min_batch"] = args.min_batch
    meta["opt_batch"] = args.opt_batch
    meta["max_batch"] = args.max_batch
    out_meta.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"[build_engine] wrote {engine_path}")
    print(f"[build_engine] wrote {out_meta}")


if __name__ == "__main__":
    main()
