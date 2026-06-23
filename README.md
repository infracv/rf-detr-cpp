# rf-detr-cpp

C++/TensorRT inference library for [RF-DETR](https://github.com/roboflow/rf-detr),
covering detection and segmentation. Targets RTX desktop GPUs and Jetson
(Orin NX, AGX Thor).

Early scaffolding — see [Implementation.md](./Implementation.md) for the full plan.

## Status

| Step | Description                                | Status   |
|------|--------------------------------------------|----------|
| 0    | Project skeleton                           | done     |
| 1    | ONNX export + engine build scripts         | done     |
| 2    | TrtSession (deserialize + IO)              | done     |
| 2.5  | C++ ONNX -> engine builder (rfdetr_build)  | done     |
| 3    | CUDA preprocess kernel                     | done     |
| 4    | Detection postprocess                      | done     |
| 5    | RFDetrDetector end-to-end                  | done     |
| 6    | Async streams + CUDA Graph                 | done*    |
| 7    | Video / camera apps                        | done     |
| 8    | Batch + dynamic shape                      | done     |
| 9    | INT8 calibration                           | done     |
| 10   | Segmenter (Mask2Former-style)              | done     |
| 11   | C ABI                                      | done     |
| 12   | Docker + Jetson                            | done     |
| 13   | Benchmark harness                          | done     |
| 14   | Polish                                     | done     |

## Build options

| CMake option              | Default | Description                            |
|---------------------------|---------|----------------------------------------|
| `RFDETR_BUILD_APPS`       | ON      | CLI binaries under `apps/`             |
| `RFDETR_BUILD_BENCHMARKS` | OFF     | Benchmark harness (`rfdetr_bench`)     |
| `RFDETR_BUILD_C_API`      | OFF     | C ABI shared library (`librfdetr_c`)   |
| `RFDETR_BUILD_SHARED`     | ON      | Build as shared library (vs static)    |
| `RFDETR_USE_OPENCV_CUDA`  | OFF     | Use OpenCV CUDA module for GPU ops     |

## Requirements

- CUDA 12.x
- TensorRT 10.x or 11.x
- OpenCV 4.5+
- CMake 3.20+
- C++17 compiler

## Build

```sh
cmake -B build -S . -DTENSORRT_DIR=/path/to/TensorRT
cmake --build build -j
```

Useful options:

- `-DCMAKE_CUDA_ARCHITECTURES=87` — Orin NX only
- `-DCMAKE_CUDA_ARCHITECTURES=101` — AGX Thor only
- `-DCMAKE_CUDA_ARCHITECTURES=120` — RTX 5070 Ti / Blackwell only
- `-DRFDETR_BUILD_APPS=ON` (default) — build CLI tools under `apps/`

## Quick start

```sh
# 0. (one-time) Python deps for ONNX export. C++ runtime needs none.
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 1. Export RF-DETR PyTorch -> ONNX (FP32). Pretrained weights cache to
#    ~/.roboflow/models/<variant>.pth (~350 MB nano) on first run.
python trt-files/scripts/export_onnx.py --variant nano --out-dir trt-files/onnx

# 2. Build a TensorRT FP32 engine. (Pure C++; no Python needed.)
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx
#     -> trt-files/onnx/rf-detr-nano-fp32.engine

# 3. Inspect from C++.
./build/rfdetr_inspect trt-files/onnx/rf-detr-nano-fp32.engine

# 4. Smoke test — runs a forward pass with zeros, prints timing + output stats.
./build/rfdetr_smoke trt-files/onnx/rf-detr-nano-fp32.engine 10

# 5. Run on an image.
./build/rfdetr_detect --engine trt-files/onnx/rf-detr-nano-fp32.engine \
                      --image test_img.jpg --out out/result.jpg

# 6. Run on a video or camera.
./build/rfdetr_video --engine trt-files/onnx/rf-detr-nano-fp32.engine \
                     --input test_video.mp4 --out out/result.mp4

# 7. Batch inference (dynamic-batch engine required for batch > 1).
./build/rfdetr_batch --engine trt-files/onnx/rf-detr-nano-fp32.engine \
                     --image test_img.jpg --batch 1

# 8. Segmentation (requires a seg-* variant engine).
python trt-files/scripts/export_onnx.py --variant seg-nano --out-dir trt-files/onnx
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-seg-nano.onnx
./build/rfdetr_seg --engine trt-files/onnx/rf-detr-seg-nano-fp32.engine \
                   --image test_img.jpg --out out/seg_result.jpg
./build/rfdetr_seg --engine trt-files/onnx/rf-detr-seg-nano-fp32.engine \
                   --video test_video.mp4 --out out/seg_video.mp4

# 9. INT8 (TRT < 11 only via rfdetr_build; any TRT via build_engine.py + trtexec).
python trt-files/scripts/int8_calibrate.py \
    --onnx   trt-files/onnx/rf-detr-nano.onnx \
    --images /path/to/calib_images \
    --output trt-files/onnx/rf-detr-nano.calib
# TRT < 11:
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx \
                     --precision int8 \
                     --calib trt-files/onnx/rf-detr-nano.calib
# Any TRT via trtexec wrapper:
python trt-files/scripts/build_engine.py \
    --onnx trt-files/onnx/rf-detr-nano.onnx \
    --precision int8 \
    --int8-calib trt-files/onnx/rf-detr-nano.calib
```

> \* CUDA Graph capture falls back to normal inference when the engine has
> data-dependent internal shapes (RF-DETR's cross-attention). The API and
> fallback path are fully implemented; the graph captures on architecturally
> simpler TRT engines.

## Benchmark

Build the benchmark binary with `-DRFDETR_BUILD_BENCHMARKS=ON`:

```sh
cmake -B build -S . -DRFDETR_BUILD_BENCHMARKS=ON -DTENSORRT_DIR=/path/to/TensorRT
cmake --build build -j
```

Run a single engine:

```sh
./build/rfdetr_bench \
    --engine  trt-files/onnx/rf-detr-nano-fp32.engine \
    --image   test_img.jpg \
    --device  rtx-5070ti \
    --warmup  50 \
    --iters   500 \
    --report  benchmarks/results/rtx5070ti_nano_fp32.json
```

Run the full matrix (all `*.engine` files under `trt-files/onnx/`):

```sh
./benchmarks/scripts/run_all.sh \
    --engine-dir trt-files/onnx \
    --image      test_img.jpg \
    --device     rtx-5070ti \
    --out-dir    benchmarks/results
```

Compare results across devices:

```sh
python benchmarks/scripts/compare_devices.py benchmarks/results/
```

### JSON schema

Each `rfdetr_bench` run writes one JSON file:

```json
{
  "device": "rtx-5070ti",
  "variant": "nano",
  "precision": "fp32",
  "task": "detection",
  "batch": 1,
  "trt_version": "10.9.0",
  "cuda_version": "12.8",
  "warmup_iters": 50,
  "measure_iters": 500,
  "latency_ms": { "mean": 4.2, "p50": 4.1, "p90": 4.5, "p99": 5.1, "min": 3.9, "max": 6.7 },
  "gpu_ms":     { "mean": 3.8, "p50": 3.7, "p90": 4.1, "p99": 4.8, "min": 3.5, "max": 5.9 },
  "throughput_fps": 238.0,
  "map_50_95": null,
  "map_50": null
}
```

## Docker

```sh
# x86_64 dev environment (GPU required):
docker build -f docker/Dockerfile.x86_64 --target dev -t rfdetr-cpp:dev .
docker run --gpus all --rm -it \
    -v $(pwd)/trt-files:/workspace/trt-files \
    -v $(pwd)/out:/workspace/out \
    rfdetr-cpp:dev bash

# Jetson (on-device):
docker build -f docker/Dockerfile.jetson -t rfdetr-cpp:jetson .
docker run --runtime nvidia --rm -it rfdetr-cpp:jetson bash
```

See [docs/jetson.md](docs/jetson.md) for Jetson-specific build and deployment instructions.

## C ABI

Build with `-DRFDETR_BUILD_C_API=ON` to produce `librfdetr_c.so`.
The header `include/rfdetr/c_api.h` is stable and suitable for Python ctypes,
Rust FFI, or any other language with a C FFI layer.

```c
#include "rfdetr/c_api.h"

rfdetr_detector_t* det = rfdetr_detector_create("model.engine", NULL);
rfdetr_detections_t* res = rfdetr_detector_detect(
    det, bgr_data, width, height, width * 3, 0.5f);
for (int i = 0; i < res->count; ++i)
    printf("cls=%d score=%.2f\n", res->detections[i].class_id, res->detections[i].score);
rfdetr_detections_free(res);
rfdetr_detector_destroy(det);
```

## License

Apache-2.0. See [LICENSE](./LICENSE).
