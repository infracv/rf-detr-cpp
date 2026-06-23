# trt-files

Scripts and outputs for the RF-DETR PyTorch → ONNX → TensorRT engine pipeline.

## Layout

- `scripts/export_onnx.py` — wraps `rfdetr.<Variant>().export(format="onnx", ...)` with sane
  defaults; writes `<basename>.onnx` and a `<basename>.json` meta sidecar consumed by
  `rfdetr/core/engine_meta.hpp`.
- `scripts/build_engine.py` — wraps `trtexec` to produce a `<basename>.engine` and copy
  the sidecar across (FP32 / FP16 / INT8). Handles dynamic-batch optimization profiles.
- `scripts/verify_engine.py` — deserializes a built engine via TRT Python bindings and
  cross-checks bindings against the sidecar.
- `onnx/` — generated ONNX + sidecar (gitignored).
- `engines/` — generated `.engine` + sidecar + INT8 `.calib` cache (gitignored).

## Pipeline

```sh
# 0. (one-time) Python dependencies for the export step. C++ side has none.
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r scripts/requirements.txt

# 1. PyTorch -> ONNX. Pretrained weights cache to ~/.roboflow/models/<variant>.pth
#    on first run (~350 MB for nano). Subsequent runs reuse the cache.
python scripts/export_onnx.py --variant small --out-dir onnx

# 2. ONNX -> TRT engine (trtexec must be on PATH; on Jetson it's at /usr/src/tensorrt/bin).
python scripts/build_engine.py \
    --onnx onnx/rf-detr-small.onnx \
    --precision fp16

# 3. Sanity-check.
python scripts/verify_engine.py --engine onnx/rf-detr-small.engine

# 4. C++ inspect (after `cmake --build build`).
../build/rfdetr_inspect onnx/rf-detr-small.engine onnx/rf-detr-small.json
```

## Sidecar schema

```json
{
  "variant": "small",
  "input_h": 512, "input_w": 512,
  "num_queries": 300,
  "num_classes": 90,
  "has_masks": false,
  "mask_h": 0, "mask_w": 0,
  "mean": [0.485, 0.456, 0.406],
  "std":  [0.229, 0.224, 0.225],
  "color_order": "RGB",
  "precision": "fp16",
  "patch_size": 16,
  "dynamic_batch": false,
  "min_batch": 1, "opt_batch": 1, "max_batch": 1
}
```

The sidecar is the authoritative description of how to feed the engine: which color
order, which mean/std, what HxW. The C++ runtime uses it instead of fragile
shape-based variant autodetection.

## Notes

- INT8 calibration is part of step 9; until then `--precision int8` requires a
  pre-existing `.calib` cache passed via `--int8-calib`.
- On Jetson, `trtexec` is usually at `/usr/src/tensorrt/bin/trtexec` — pass
  `--trtexec /usr/src/tensorrt/bin/trtexec` to `build_engine.py` if it isn't on PATH.
