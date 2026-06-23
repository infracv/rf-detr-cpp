#!/usr/bin/env python3
"""INT8 calibration for RF-DETR TensorRT engines.

Runs TRT's IInt8EntropyCalibrator2 over a directory of calibration images
and writes a `.calib` cache file.  That cache is then consumed by
`rfdetr_build --precision int8 --calib <cache>` or
`build_engine.py --precision int8 --int8-calib <cache>`.

Requirements (beyond requirements.txt):
    tensorrt   -- already installed with TRT (pip install tensorrt)

Example:
    # Export ONNX first (one-time):
    python trt-files/scripts/export_onnx.py --variant nano --out-dir trt-files/onnx

    # Calibrate (needs ~100-500 representative images):
    python trt-files/scripts/int8_calibrate.py \\
        --onnx    trt-files/onnx/rf-detr-nano.onnx \\
        --images  /path/to/calib_images/ \\
        --output  trt-files/onnx/rf-detr-nano.calib

    # Build INT8 engine via C++ builder:
    ./build/rfdetr_build \\
        --onnx      trt-files/onnx/rf-detr-nano.onnx \\
        --precision int8 \\
        --calib     trt-files/onnx/rf-detr-nano.calib

    # OR build via Python trtexec wrapper:
    python trt-files/scripts/build_engine.py \\
        --onnx        trt-files/onnx/rf-detr-nano.onnx \\
        --precision   int8 \\
        --int8-calib  trt-files/onnx/rf-detr-nano.calib

Notes:
  - Use 200-500 representative images (diverse scene content, not just COCO val).
  - Images are resized to the model's square input resolution using the same
    square-resize + ImageNet normalization pipeline as the C++ runtime.
  - The .calib file is GPU-architecture-specific; rebuild after changing the GPU.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Preprocessing: replicates src/core/cuda_preprocess.cu on CPU (numpy)
# ---------------------------------------------------------------------------

def preprocess_image(img_bgr, R: int, mean, std) -> np.ndarray:
    """Square resize + BGR->RGB + /255 + ImageNet normalize -> CHW float32."""
    import cv2
    img = cv2.resize(img_bgr, (R, R), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0  # BGR->RGB, /255
    img = (img - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return img.transpose(2, 0, 1).astype(np.float32)  # HWC->CHW


def collect_image_paths(images_dir: Path, max_images: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)
    if not paths:
        sys.exit(f"error: no images found in {images_dir}")
    if len(paths) > max_images:
        # Evenly sample to stay under the limit
        step = len(paths) / max_images
        paths = [paths[int(i * step)] for i in range(max_images)]
    return paths


# ---------------------------------------------------------------------------
# TRT INT8 Calibrator
# ---------------------------------------------------------------------------

class RFDetrCalibrator:
    """IInt8EntropyCalibrator2 implementation for RF-DETR."""

    def __init__(self, image_paths, R, mean, std, batch_size, cache_path):
        import tensorrt as trt
        self._trt        = trt
        self._paths      = image_paths
        self._R          = R
        self._mean       = mean
        self._std        = std
        self._batch      = batch_size
        self._cache_path = Path(cache_path)
        self._cursor     = 0
        self._device_buf = None

        # Allocate a pinned host buffer + device buffer for one batch.
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        self._nbytes = batch_size * 3 * R * R * 4  # float32
        self._host   = cuda.pagelocked_empty(
            (batch_size, 3, R, R), dtype=np.float32)
        self._device = cuda.mem_alloc(self._nbytes)

    # IInt8Calibrator interface ------------------------------------------------

    def get_batch_size(self):
        return self._batch

    def get_batch(self, names):
        import pycuda.driver as cuda

        remaining = len(self._paths) - self._cursor
        if remaining <= 0:
            return None

        actual = min(self._batch, remaining)
        for i in range(actual):
            path = self._paths[self._cursor + i]
            import cv2
            img = cv2.imread(str(path))
            if img is None:
                print(f"[calibrate] warning: could not read {path}, using zeros")
                self._host[i] = 0.0
            else:
                self._host[i] = preprocess_image(img, self._R, self._mean, self._std)
        # Pad remaining slots in partial last batch with zeros.
        for i in range(actual, self._batch):
            self._host[i] = 0.0

        cuda.memcpy_htod(self._device, self._host)
        self._cursor += actual
        print(f"\r[calibrate] {min(self._cursor, len(self._paths))}/{len(self._paths)} images",
              end="", flush=True)
        return [int(self._device)]

    def read_calibration_cache(self):
        if self._cache_path.exists():
            print(f"[calibrate] loading existing cache: {self._cache_path}")
            return self._cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self._cache_path.write_bytes(cache)
        print(f"\n[calibrate] wrote calibration cache: {self._cache_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx",    required=True,
                    help="Input ONNX model path")
    ap.add_argument("--images",  required=True,
                    help="Directory of calibration images (JPEG/PNG)")
    ap.add_argument("--output",  default=None,
                    help="Output .calib cache path (default: <onnx-stem>.calib)")
    ap.add_argument("--meta",    default=None,
                    help="Sidecar JSON (default: <onnx>.json)")
    ap.add_argument("--max-images", type=int, default=500,
                    help="Maximum calibration images to use (default: 500)")
    ap.add_argument("--batch",   type=int, default=1,
                    help="Calibration batch size (default: 1)")
    ap.add_argument("--workspace-mib", type=int, default=4096,
                    help="TRT workspace in MiB (default: 4096)")
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        sys.exit(f"error: onnx not found: {onnx_path}")

    meta_path = Path(args.meta) if args.meta else onnx_path.with_suffix(".json")
    if not meta_path.is_file():
        sys.exit(f"error: meta sidecar not found: {meta_path}\n"
                 f"  run export_onnx.py first, or pass --meta")
    meta = json.loads(meta_path.read_text())

    output = Path(args.output) if args.output else onnx_path.with_suffix(".calib")

    images_dir = Path(args.images)
    if not images_dir.is_dir():
        sys.exit(f"error: images directory not found: {images_dir}")

    R    = meta["input_h"]
    mean = meta["mean"]
    std  = meta["std"]

    print(f"[calibrate] model:        {onnx_path.name}")
    print(f"[calibrate] input size:   {R}x{R}")
    print(f"[calibrate] images dir:   {images_dir}")
    print(f"[calibrate] output cache: {output}")

    paths = collect_image_paths(images_dir, args.max_images)
    print(f"[calibrate] images found: {len(paths)} (max={args.max_images})")

    try:
        import tensorrt as trt
    except ImportError:
        sys.exit("error: tensorrt Python package not installed\n"
                 "  pip install tensorrt  (or install via TRT wheel)")

    try:
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda  # noqa: F401
    except ImportError:
        sys.exit("error: pycuda not installed\n"
                 "  pip install pycuda")

    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.exit("error: opencv-python not installed\n"
                 "  pip install opencv-python")

    logger = trt.Logger(trt.Logger.WARNING)
    calibrator = RFDetrCalibrator(paths, R, mean, std, args.batch, output)

    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser  = trt.OnnxParser(network, logger)

    print(f"[calibrate] parsing ONNX ...")
    with open(str(onnx_path), "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX error: {parser.get_error(i).desc()}")
            sys.exit("error: ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,
                                 args.workspace_mib * (1 << 20))
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_flag(trt.BuilderFlag.FP16)  # FP16 fallback for layers that can't be INT8
    config.int8_calibrator = calibrator

    print(f"[calibrate] building calibration engine (this may take 1-5 min) ...")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        sys.exit("error: engine build failed during calibration")

    print(f"[calibrate] done — cache written to {output}")
    print(f"[calibrate] use with:")
    print(f"  ./build/rfdetr_build --onnx {onnx_path} --precision int8 --calib {output}")


if __name__ == "__main__":
    main()
