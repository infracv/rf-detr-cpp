#!/usr/bin/env python3
"""convert_int8.py — Create a QDQ (Quantize-Dequantize) INT8 ONNX for RF-DETR.

Uses onnxruntime.quantization.quantize_static() with a representative
calibration dataset to insert per-tensor Q/DQ nodes.  The resulting ONNX
is compatible with TensorRT 11+ which uses strongly-typed networks: TRT
reads the QDQ nodes and generates an INT8 engine without requiring the
legacy --int8 / calibration-cache flags.

For TRT < 11 use the legacy implicit-quantization path instead:
    python trt-files/scripts/int8_calibrate.py \\
        --onnx <model.onnx> --images <dir> --output <model.calib>
    rfdetr_build --onnx <model.onnx> --precision int8 --calib <model.calib>

Requirements:
    pip install onnxruntime onnxruntime-tools opencv-python onnx

Usage:
    python trt-files/scripts/convert_int8.py \\
        --onnx   trt-files/onnx/rf-detr-nano.onnx \\
        --images /path/to/calib_images/ \\
        --out    trt-files/onnx/rf-detr-nano-int8.onnx

Then build the TRT engine (TRT 11+):
    rfdetr_build --onnx trt-files/onnx/rf-detr-nano-int8.onnx --precision fp32
    # engine runs INT8 internally because TRT reads the embedded QDQ nodes

Notes:
  - 200-500 representative images from your deployment domain give best accuracy.
  - Preprocessing replicates the C++ CUDA pipeline exactly:
    square resize → BGR→RGB → /255 → ImageNet normalize.
  - Graph inputs/outputs remain FP32 so the C++ runtime is unchanged.
  - Some transformer operations (LayerNorm, Softmax, relative-position bias)
    are excluded from INT8 quantization by default; use --no-exclude-sensitive
    to quantize everything (may reduce accuracy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Preprocessing — must match src/core/cuda_preprocess.cu exactly
# ---------------------------------------------------------------------------

def preprocess_image(img_bgr, R: int, mean, std) -> np.ndarray:
    """Square resize + BGR→RGB + /255 + ImageNet normalize → CHW float32."""
    import cv2
    img = cv2.resize(img_bgr, (R, R), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    img = (img - mean) / std
    return img.transpose(2, 0, 1).astype(np.float32)


def collect_image_paths(images_dir: Path, max_images: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)
    if not paths:
        sys.exit(f"error: no images found in {images_dir}")
    if len(paths) > max_images:
        step = len(paths) / max_images
        paths = [paths[int(i * step)] for i in range(max_images)]
    return paths


# ---------------------------------------------------------------------------
# CalibrationDataReader
# ---------------------------------------------------------------------------

class RFDetrCalibDataReader:
    """onnxruntime CalibrationDataReader for RF-DETR.

    Yields batches of pre-processed images with the same pipeline as the C++
    CUDA preprocessor so activation statistics match the deployment distribution.
    """

    def __init__(self, image_paths: list[Path], R: int, mean, std,
                 input_name: str = "input", batch_size: int = 1):
        self._paths      = image_paths
        self._R          = R
        self._mean       = np.array(mean, dtype=np.float32)
        self._std        = np.array(std,  dtype=np.float32)
        self._name       = input_name
        self._batch      = batch_size
        self._cursor     = 0

    def get_next(self):
        import cv2
        if self._cursor >= len(self._paths):
            return None
        end = min(self._cursor + self._batch, len(self._paths))
        batch = []
        for path in self._paths[self._cursor:end]:
            img = cv2.imread(str(path))
            if img is None:
                print(f"\n[convert_int8] warning: cannot read {path}, using zeros")
                chw = np.zeros((3, self._R, self._R), dtype=np.float32)
            else:
                chw = preprocess_image(img, self._R, self._mean, self._std)
            batch.append(chw)
        self._cursor = end
        print(f"\r[convert_int8] calibrating {self._cursor}/{len(self._paths)}",
              end="", flush=True)
        return {self._name: np.stack(batch, axis=0)}


# ---------------------------------------------------------------------------
# Sensitive op detection — ops that hurt accuracy when INT8-quantized
# ---------------------------------------------------------------------------

# Ops whose outputs change a lot with INT8; exclude them so they stay FP32.
SENSITIVE_OP_TYPES = {
    "LayerNormalization",
    "SkipLayerNormalization",
    "EmbedLayerNormalization",
    "Softmax",
    "Gelu",
    "BiasGelu",
    "FastGelu",
}


def find_sensitive_nodes(onnx_path: Path) -> list[str]:
    """Return node names of known-sensitive ops and scalar Constant nodes.

    TRT 11 rejects per-channel QDQ nodes on 0-D/scalar tensors (axis=1 is
    out of range for a rank-0 tensor).  onnxruntime.quantization inserts such
    nodes when it quantizes Constant ops that produce scalars (e.g. GELU
    multipliers).  Excluding them keeps those constants as FP32 and avoids
    the TRT parse error.
    """
    import onnx
    from onnx import numpy_helper
    import numpy as np
    model = onnx.load(str(onnx_path))
    names = []
    for node in model.graph.node:
        # Accuracy-sensitive op types.
        if node.op_type in SENSITIVE_OP_TYPES and node.name:
            names.append(node.name)
            continue
        # Scalar / 0-D Constant nodes — TRT rejects QDQ with axis=1 on them.
        if node.op_type == "Constant" and node.name:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.TENSOR:
                    arr = numpy_helper.to_array(attr.t)
                    if arr.ndim <= 1:  # 0-D scalar or 1-D length-1
                        names.append(node.name)
                    break
    return names


# ---------------------------------------------------------------------------
# Post-processing: strip QDQ nodes that TRT 11 cannot parse
# ---------------------------------------------------------------------------

def _strip_invalid_qdq(model) -> object:
    """Remove QuantizeLinear/DequantizeLinear pairs that TRT 11 rejects.

    TRT 11 errors when a QDQ node has axis=1 but the quantized tensor is
    0-D or 1-D (e.g. scalar GELU multipliers that onnxruntime quantized with
    per-channel settings).  We find such nodes by checking the scale tensor
    shape: if the scale is a vector (not a scalar), TRT will apply per-channel
    quantization and reject 0-D inputs.

    Strategy: for each bad QuantizeLinear, reroute its consumer(s) to use its
    un-quantized input; remove both the QuantizeLinear and its paired
    DequantizeLinear from the graph.
    """
    import onnx
    from onnx import numpy_helper, TensorProto

    graph = model.graph

    # Build a map: tensor_name → its shape (from initializers + value_info + inputs)
    shape_map = {}
    for init in graph.initializer:
        shape_map[init.name] = list(init.dims)
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        dims = vi.type.tensor_type.shape
        if dims is not None:
            shape_map[vi.name] = [d.dim_value for d in dims.dim]

    # Index nodes by output name for quick lookup.
    node_by_output = {}
    for node in graph.node:
        for out in node.output:
            node_by_output[out] = node

    def _tensor_is_scalar_or_low_rank(name: str) -> bool:
        """Return True if the named tensor is 0-D or 1-D with length ≤ 1."""
        shape = shape_map.get(name)
        if shape is None:
            return False
        if len(shape) == 0:
            return True
        if len(shape) == 1 and (shape[0] == 0 or shape[0] == 1):
            return True
        return False

    def _scale_is_vector(name: str) -> bool:
        """Return True if the scale tensor is a non-scalar (per-channel)."""
        shape = shape_map.get(name)
        if shape is None:
            # Try initializer directly.
            for init in graph.initializer:
                if init.name == name:
                    return len(init.dims) >= 1 and (init.dims[0] if init.dims else 1) > 1
            return False
        return len(shape) >= 1 and (shape[0] if shape else 1) > 1

    bad_q_outputs = set()   # outputs of QuantizeLinear to remove
    bad_dq_outputs = set()  # outputs of DequantizeLinear to remove
    reroute = {}            # dq_output → original_input_of_q

    for node in graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        if len(node.input) < 1:
            continue
        x_name   = node.input[0]
        q_output = node.output[0]

        # TRT 11 rejects QDQ on 0-D/scalar inputs regardless of scale shape.
        if not _tensor_is_scalar_or_low_rank(x_name):
            continue

        # This Q node is invalid for TRT.  Find its paired DQ node.
        dq_node = node_by_output.get(q_output + "_DequantizeLinear_Output") or None
        if dq_node is None:
            # DQ output name follows ORT naming: <q_input>_QuantizeLinear_Output
            # fed into DQ whose output is <orig>_DequantizeLinear_Output.
            # Search for DQ that takes q_output as input.
            for n in graph.node:
                if n.op_type == "DequantizeLinear" and n.input and n.input[0] == q_output:
                    dq_node = n
                    break

        bad_q_outputs.add(q_output)
        if dq_node is not None:
            for dq_out in dq_node.output:
                bad_dq_outputs.add(dq_out)
                reroute[dq_out] = x_name  # bypass: consumers get original input

    if not bad_q_outputs and not bad_dq_outputs:
        return model  # nothing to do

    # Rewrite all nodes: remove bad Q/DQ nodes, patch inputs that used DQ outputs.
    new_nodes = []
    removed = 0
    for node in graph.node:
        if node.op_type == "QuantizeLinear" and any(o in bad_q_outputs for o in node.output):
            removed += 1
            continue
        if node.op_type == "DequantizeLinear" and any(o in bad_dq_outputs for o in node.output):
            removed += 1
            continue
        # Patch inputs of surviving nodes.
        new_inputs = [reroute.get(inp, inp) for inp in node.input]
        if new_inputs != list(node.input):
            del node.input[:]
            node.input.extend(new_inputs)
        new_nodes.append(node)

    del graph.node[:]
    graph.node.extend(new_nodes)
    print(f"  removed {removed} invalid QDQ nodes (scalar/0-D per-channel)")
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx",    required=True,
                    help="Input FP32 ONNX model (from export_onnx.py)")
    ap.add_argument("--images",  required=True,
                    help="Directory of calibration images (JPEG/PNG/BMP)")
    ap.add_argument("--out",     default=None,
                    help="Output QDQ ONNX path (default: <stem>-int8.onnx)")
    ap.add_argument("--meta",    default=None,
                    help="Input meta sidecar (default: <onnx>.json)")
    ap.add_argument("--max-images",   type=int, default=500,
                    help="Max calibration images (default: 500)")
    ap.add_argument("--batch",        type=int, default=1,
                    help="Calibration batch size (default: 1)")
    ap.add_argument("--per-channel",  action="store_true",
                    help="Use per-channel weight quantization (slower build, may be more accurate)")
    ap.add_argument("--no-exclude-sensitive", action="store_true",
                    help="Quantize LayerNorm/Softmax/etc (may reduce accuracy)")
    ap.add_argument("--input-name",   default="input",
                    help="ONNX graph input tensor name (default: input)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx).resolve()
    if not onnx_path.is_file():
        sys.exit(f"error: ONNX not found: {onnx_path}")

    out_path = Path(args.out) if args.out else onnx_path.parent / (onnx_path.stem + "-int8.onnx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta_path = Path(args.meta) if args.meta else onnx_path.with_suffix(".json")
    if not meta_path.is_file():
        sys.exit(f"error: meta sidecar not found: {meta_path}\n"
                 f"  run export_onnx.py first, or pass --meta")
    meta = json.loads(meta_path.read_text())

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        sys.exit(f"error: images directory not found: {images_dir}")

    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.exit("error: opencv-python not installed\n  pip install opencv-python")

    try:
        from onnxruntime.quantization import (
            QuantFormat, QuantType, quantize_static,
        )
    except ImportError:
        sys.exit("error: onnxruntime.quantization not available\n"
                 "  pip install onnxruntime onnxruntime-tools")

    R    = meta["input_h"]
    mean = meta["mean"]
    std  = meta["std"]

    print(f"[convert_int8] model:         {onnx_path.name}")
    print(f"[convert_int8] input size:    {R}x{R}")
    print(f"[convert_int8] images dir:    {images_dir}")
    print(f"[convert_int8] output:        {out_path}")

    paths = collect_image_paths(images_dir, args.max_images)
    print(f"[convert_int8] calib images:  {len(paths)} (max={args.max_images})")

    # Nodes to exclude from INT8 (accuracy-sensitive ops).
    nodes_to_exclude = []
    if not args.no_exclude_sensitive:
        nodes_to_exclude = find_sensitive_nodes(onnx_path)
        if nodes_to_exclude:
            print(f"[convert_int8] excluding {len(nodes_to_exclude)} nodes "
                  f"(LayerNorm/Softmax/Gelu + scalar Constants — TRT 11 compatibility)")

    reader = RFDetrCalibDataReader(paths, R, mean, std,
                                    input_name=args.input_name,
                                    batch_size=args.batch)

    print(f"[convert_int8] running static quantization (QDQ format) ...")
    quantize_static(
        model_input=str(onnx_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=args.per_channel,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
        nodes_to_exclude=nodes_to_exclude,
        # Keep graph I/O as FP32 so the C++ runtime buffers are unchanged.
        extra_options={
            # TRT requires symmetric quantization (zero_point = 0).
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            # TRT 11 rejects INT32 DequantizeLinear (biases are INT32 by ONNX
            # spec but TRT only accepts INT8/FP8/INT4 for DQ inputs).
            "QuantizeBias": False,
        },
    )
    print()  # newline after the \r progress

    # Post-process: strip invalid per-channel QDQ nodes that TRT 11 rejects.
    # onnxruntime may insert QuantizeLinear/DequantizeLinear with axis=1 on
    # 0-D scalar tensors (e.g. GELU multipliers).  TRT errors: "axis must be
    # in range [0, nbDims(0)]".  We remove those Q/DQ pairs and reconnect
    # the graph to bypass them.
    import onnx
    model = onnx.load(str(out_path))
    model = _strip_invalid_qdq(model)
    onnx.save(model, str(out_path))
    print(f"[convert_int8] stripped invalid QDQ nodes (TRT 11 compatibility)")

    # Validate the output.
    try:
        onnx.checker.check_model(model, full_check=False)
        print("[convert_int8] ONNX check: OK")
    except Exception as e:
        print(f"[convert_int8] ONNX check warning: {e} (may still build with TRT)")

    # Write meta sidecar with precision=int8.
    out_meta = out_path.with_suffix(".json")
    meta_out = dict(meta)
    meta_out["precision"] = "int8"
    out_meta.write_text(json.dumps(meta_out, indent=2) + "\n")
    print(f"[convert_int8] meta:          {out_meta}")

    print("[convert_int8] done.")
    print()
    print("Next step — build TRT engine (TRT 11+):")
    print(f"  rfdetr_build --onnx {out_path} --precision fp32")
    print(f"  # TRT reads the embedded QDQ nodes and generates INT8 kernels")
    print()
    print("Or for TRT < 11 (implicit quantization), use instead:")
    print(f"  python trt-files/scripts/int8_calibrate.py \\")
    print(f"      --onnx {onnx_path} --images {images_dir} \\")
    print(f"      --output {onnx_path.parent / (onnx_path.stem + '.calib')}")
    print(f"  rfdetr_build --onnx {onnx_path} --precision int8 \\")
    print(f"      --calib {onnx_path.parent / (onnx_path.stem + '.calib')}")


if __name__ == "__main__":
    main()
