#!/usr/bin/env python3
"""Export an RF-DETR PyTorch checkpoint to ONNX, plus a meta-sidecar JSON.

The sidecar is consumed by `rfdetr/core/engine_meta.hpp`. It captures the
variant identity, input H/W, query count, normalization stats, and color
order — i.e. everything the C++ runtime needs that is NOT recoverable from the
engine's tensor shapes alone.

Example:
    python trt-files/scripts/export_onnx.py --variant small --out-dir trt-files/onnx
    python trt-files/scripts/export_onnx.py --variant seg-large --weights /tmp/seg-large.pth
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict


# Source of truth: src/rfdetr/variants.py + src/rfdetr/config.py in roboflow/rf-detr.
# See Implementation.md §1.1 for the full table.
VARIANT_TABLE: Dict[str, Dict[str, Any]] = {
    "nano":        {"resolution": 384, "num_queries": 300, "patch": 16, "has_masks": False,
                    "ctors": ("RFDETRNano",)},
    "small":       {"resolution": 512, "num_queries": 300, "patch": 16, "has_masks": False,
                    "ctors": ("RFDETRSmall",)},
    "medium":      {"resolution": 576, "num_queries": 300, "patch": 16, "has_masks": False,
                    "ctors": ("RFDETRMedium",)},
    "base":        {"resolution": 560, "num_queries": 300, "patch": 14, "has_masks": False,
                    "ctors": ("RFDETRBase",)},
    "large":       {"resolution": 704, "num_queries": 300, "patch": 16, "has_masks": False,
                    "ctors": ("RFDETRLarge",)},
    "seg-nano":    {"resolution": 312, "num_queries": 100, "patch": 12, "has_masks": True,
                    "ctors": ("RFDETRSegPreview", "RFDETRSegNano")},
    "seg-small":   {"resolution": 384, "num_queries": 100, "patch": 12, "has_masks": True,
                    "ctors": ("RFDETRSegSmall",)},
    "seg-medium":  {"resolution": 432, "num_queries": 200, "patch": 12, "has_masks": True,
                    "ctors": ("RFDETRSegMedium",)},
    "seg-large":   {"resolution": 504, "num_queries": 200, "patch": 12, "has_masks": True,
                    "ctors": ("RFDETRSegLarge",)},
    "seg-xl":      {"resolution": 624, "num_queries": 300, "patch": 12, "has_masks": True,
                    "ctors": ("RFDETRSegXL",)},
    "seg-2xl":     {"resolution": 768, "num_queries": 300, "patch": 12, "has_masks": True,
                    "ctors": ("RFDETRSeg2XL",)},
    "seg-preview": {"resolution": 432, "num_queries": 200, "patch": 12, "has_masks": True,
                    "ctors": ("RFDETRSegPreview",)},
}


def resolve_ctor(rfdetr_module, ctor_names):
    for name in ctor_names:
        if hasattr(rfdetr_module, name):
            return getattr(rfdetr_module, name)
    raise AttributeError(
        f"none of {ctor_names} are exposed by `rfdetr` — upstream may have renamed the variant. "
        f"Edit VARIANT_TABLE in this script to match."
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANT_TABLE.keys()),
                    help="RF-DETR variant to export")
    ap.add_argument("--weights", default=None,
                    help="path to .pth checkpoint (default: upstream pretrained)")
    ap.add_argument("--out-dir", default="trt-files/onnx",
                    help="output directory for .onnx and .json sidecar")
    ap.add_argument("--name", default=None,
                    help="output basename (default: rf-detr-<variant>)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--dynamic-batch", action="store_true",
                    help="export with batch dimension as a dynamic axis")
    args = ap.parse_args()

    spec = VARIANT_TABLE[args.variant]

    try:
        import rfdetr  # type: ignore
    except ImportError:
        sys.exit("rfdetr Python package not installed (pip install rfdetr)")

    ctor = resolve_ctor(rfdetr, spec["ctors"])
    model_kwargs = {"pretrain_weights": args.weights} if args.weights else {}
    model = ctor(**model_kwargs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    basename = args.name or f"rf-detr-{args.variant}"

    # rfdetr's `.export(format="onnx", output_dir=...)` typically writes
    # `inference_model.onnx` (and possibly a sim'd variant) into output_dir.
    # We invoke it, then locate the produced .onnx and rename to our convention.
    print(f"[export_onnx] exporting variant={args.variant} to {out_dir}/")
    model.export(
        format="onnx",
        output_dir=str(out_dir),
        opset_version=args.opset,
        dynamic_batch=args.dynamic_batch,
    )

    # Find the most recently produced .onnx in out_dir; rename to <basename>.onnx.
    candidates = sorted(out_dir.glob("*.onnx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        sys.exit(f"export reported success but no .onnx file appeared in {out_dir}")
    src = candidates[0]
    dst = out_dir / f"{basename}.onnx"
    if src.resolve() != dst.resolve():
        shutil.move(str(src), str(dst))
        print(f"[export_onnx] renamed {src.name} -> {dst.name}")

    R = spec["resolution"]
    meta = {
        "variant": args.variant,
        "input_h": R,
        "input_w": R,
        "num_queries": spec["num_queries"],
        "num_classes": 90,
        "has_masks": spec["has_masks"],
        "mask_h": R // 4 if spec["has_masks"] else 0,
        "mask_w": R // 4 if spec["has_masks"] else 0,
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
        "color_order": "RGB",
        "precision": "onnx",
        "patch_size": spec["patch"],
        "dynamic_batch": args.dynamic_batch,
        "min_batch": 1,
        "opt_batch": 1,
        "max_batch": 1,
        "opset": args.opset,
        "source": "roboflow/rf-detr",
    }
    meta_path = out_dir / f"{basename}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[export_onnx] wrote {dst}")
    print(f"[export_onnx] wrote {meta_path}")


if __name__ == "__main__":
    main()
