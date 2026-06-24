<h1 align="center">RF-DETR C++</h1>
<h3 align="center">Production-Ready RF-DETR Inference Engine for C++ / TensorRT</h3>

<p align="center">
  <em>Transformer-based real-time object detection and segmentation — no Python at inference time</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-features">Features</a> ·
  <a href="#-api-reference">API</a> ·
  <a href="#-benchmarks">Benchmarks</a> ·
  <a href="#-docker">Docker</a> ·
  <a href="docs/jetson.md">Jetson Guide</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/TensorRT-%E2%89%A5%2010.0-76B900?logo=nvidia&logoColor=white" alt="TensorRT"/>
  <img src="https://img.shields.io/badge/CUDA-%E2%89%A5%2012.0-76B900?logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg?logo=cplusplus&logoColor=white" alt="C++17"/>
  <img src="https://img.shields.io/badge/license-Apache--2.0-ef4444" alt="License"/>
</p>

---

## What is RF-DETR?

[RF-DETR](https://github.com/roboflow/rf-detr) is Roboflow's transformer-based real-time object detection model built on a DINOv2 backbone. Unlike YOLO models, RF-DETR uses a DETR-style architecture that:

- **Eliminates NMS** — set prediction via cross-attention; top-K candidates are scored directly
- **Requires square input** — no letterboxing, forced resize to the model's native resolution
- **Sigmoid + top-K postprocess** — scores are raw sigmoid outputs, no softmax or NMS post-step
- **Mask2Former-style segmentation** — per-query mask logits upsampled and thresholded at inference

This library brings RF-DETR to C++/TensorRT for production deployment on RTX desktop GPUs and NVIDIA Jetson.

---

## Features

| Capability | Description |
|------------|-------------|
| **Object Detection** | All RF-DETR variants: nano, small, medium, base, large |
| **Instance Segmentation** | All seg-* variants with bilinear mask upsample + threshold |
| **CUDA Preprocessing** | Single GPU kernel: square resize + BGR→RGB + ImageNet normalize |
| **Async Inference** | Pinned memory, async H2D/D2H transfers, single CUDA stream |
| **CUDA Graph** | Captured once for fixed-shape engines; auto-fallback for dynamic |
| **Batch Inference** | Dynamic batch shape — one engine for batch 1 through N |
| **INT8 Quantization** | Python calibration script + C++ builder (TRT < 11); `trtexec` for TRT 11 |
| **C ABI** | Stable `librfdetr_c.so` — call from Python ctypes, Rust FFI, Go cgo |
| **Docker** | Multi-stage x86_64 and Jetson (JetPack 6.x) images |
| **Zero Python at runtime** | Python only needed for ONNX export and INT8 calibration |

---

## Supported Models

| Variant | Task | Input | COCO mAP |
|---------|------|-------|----------|
| `nano` | Detection | 384×384 | ~43 |
| `small` | Detection | 512×512 | ~48 |
| `medium` | Detection | 576×576 | ~51 |
| `base` | Detection | 560×560 | ~53 |
| `large` | Detection | 704×704 | ~55 |
| `seg-nano` | Detection + Segmentation | 312×312 | — |
| `seg-medium` | Detection + Segmentation | 432×432 | — |
| `seg-large` | Detection + Segmentation | 504×504 | — |

---

## Quick Start

### Prerequisites

| Dependency | Version | Notes |
|------------|---------|-------|
| NVIDIA GPU | CC ≥ 8.0 | RTX 30xx / 40xx / 50xx or Jetson Orin / AGX Thor |
| CUDA Toolkit | ≥ 12.0 | |
| TensorRT | ≥ 10.0 | TRT 11 supported (strongly-typed networks) |
| OpenCV | ≥ 4.5 | core, imgproc, imgcodecs, videoio, highgui |
| CMake | ≥ 3.20 | |
| C++ compiler | C++17 | GCC 9+, Clang 10+ |

### Build

```sh
git clone https://github.com/infracv/rf-detr-cpp.git
cd rf-detr-cpp

cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_DIR=/path/to/TensorRT
cmake --build build -j$(nproc)
```

<details>
<summary><b>Custom CUDA architectures</b></summary>

```sh
# RTX 50xx (Blackwell)
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=120

# Jetson Orin NX
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=87

# Jetson AGX Thor
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=101

# Multiple in one binary
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES="87;101;120"
```

</details>

### Export and Run

```sh
# 1. (One-time) Python deps for ONNX export. C++ runtime needs none.
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Export RF-DETR PyTorch → ONNX
python trt-files/scripts/export_onnx.py --variant nano --out-dir trt-files/onnx

# 3. Build TensorRT engine (pure C++, no Python)
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp32
#   → trt-files/onnx/rf-detr-nano-fp32.engine

# 4. Detect on an image
./build/rfdetr_detect \
    --engine trt-files/onnx/rf-detr-nano-fp32.engine \
    --image  test_img.jpg \
    --out    out/result.jpg

# 5. Run on a video
./build/rfdetr_video \
    --engine trt-files/onnx/rf-detr-nano-fp32.engine \
    --input  test_video.mp4 \
    --out    out/result.mp4

# 6. Segmentation
python trt-files/scripts/export_onnx.py --variant seg-nano --out-dir trt-files/onnx
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-seg-nano.onnx
./build/rfdetr_seg \
    --engine trt-files/onnx/rf-detr-seg-nano-fp32.engine \
    --image  test_img.jpg \
    --out    out/seg_result.jpg
```

---

## API Reference

### Object Detection

```cpp
#include "rfdetr/tasks/detector.hpp"

rfdetr::RFDetrDetector detector("rf-detr-nano-fp32.engine");

cv::Mat image = cv::imread("image.jpg");
rfdetr::Detections dets = detector.detect(image, /*threshold=*/0.5f);

for (const auto& d : dets) {
    std::printf("class=%d  score=%.2f  box=[%.1f %.1f %.1f %.1f]\n",
                d.class_id, d.score,
                d.box.x1, d.box.y1, d.box.x2, d.box.y2);
}

// Visualize
rfdetr::draw_detections(image, dets);
cv::imwrite("out.jpg", image);
```

### Instance Segmentation

```cpp
#include "rfdetr/tasks/segmenter.hpp"

rfdetr::RFDetrSegmenter seg("rf-detr-seg-nano-fp32.engine");

rfdetr::Detections dets = seg.segment(image, /*threshold=*/0.5f);

// Draw semi-transparent masks + boxes
rfdetr::draw_segmentations(image, dets);
cv::imwrite("out.jpg", image);
```

### Batch Inference

```cpp
std::vector<cv::Mat> frames = { img1, img2, img3, img4 };

rfdetr::RFDetrDetector detector("rf-detr-nano-fp32.engine");
auto batch_results = detector.detect_batch(frames, 0.5f);

for (std::size_t i = 0; i < batch_results.size(); ++i)
    std::printf("frame %zu: %zu detections\n", i, batch_results[i].size());
```

### C ABI (Python / Rust / Go FFI)

Build with `-DRFDETR_BUILD_C_API=ON` to produce `librfdetr_c.so`:

```c
#include "rfdetr/c_api.h"

rfdetr_detector_t* det = rfdetr_detector_create("model.engine", NULL);

rfdetr_detections_t* res = rfdetr_detector_detect(
    det, bgr_data, width, height, width * 3, 0.5f);

for (int i = 0; i < res->count; ++i)
    printf("cls=%d score=%.2f\n",
           res->detections[i].class_id, res->detections[i].score);

rfdetr_detections_free(res);
rfdetr_detector_destroy(det);
```

Python via ctypes:

```python
import ctypes, numpy as np, cv2

lib = ctypes.CDLL("librfdetr_c.so")
# ... (see include/rfdetr/c_api.h for full binding)
```

---

## CLI Tools

| Binary | Description |
|--------|-------------|
| `rfdetr_build` | Convert ONNX → TensorRT engine (fp32 / int8) |
| `rfdetr_detect` | Run detection on a single image |
| `rfdetr_video` | Run detection on a video file or camera stream |
| `rfdetr_batch` | Batch inference benchmark |
| `rfdetr_seg` | Run segmentation on image or video |
| `rfdetr_bench` | Latency + throughput benchmark with JSON report |
| `rfdetr_inspect` | Print engine I/O bindings and shapes |
| `rfdetr_smoke` | Quick forward-pass sanity check |

---

## Benchmarks

> **Note:** Results below are placeholders. Full benchmark runs are in progress on target hardware.

### RTX 5070 Ti — RF-DETR nano · 384×384 · FP32 · Batch 1

| Metric | Value |
|--------|-------|
| Latency mean | — ms |
| Latency p50 | — ms |
| Latency p99 | — ms |
| Throughput | — FPS |
| GPU memory | — MB |

### NVIDIA Jetson AGX Thor — RF-DETR nano · 384×384 · FP32 · Batch 1

| Metric | Value |
|--------|-------|
| Latency mean | — ms |
| Latency p50 | — ms |
| Latency p99 | — ms |
| Throughput | — FPS |
| GPU memory | — MB |

### NVIDIA Jetson Orin NX 16 GB — RF-DETR nano · 384×384 · FP32 · Batch 1

| Metric | Value |
|--------|-------|
| Latency mean | — ms |
| Latency p50 | — ms |
| Latency p99 | — ms |
| Throughput | — FPS |
| GPU memory | — MB |

<details>
<summary><b>Run your own benchmarks</b></summary>

```sh
cmake -B build -S . -DRFDETR_BUILD_BENCHMARKS=ON
cmake --build build -j

# Single engine
./build/rfdetr_bench \
    --engine  trt-files/onnx/rf-detr-nano-fp32.engine \
    --image   test_img.jpg \
    --device  rtx-5070ti \
    --warmup  50 \
    --iters   500 \
    --report  benchmarks/results/rtx5070ti_nano_fp32.json

# Full matrix (all *.engine files)
./benchmarks/scripts/run_all.sh \
    --engine-dir trt-files/onnx \
    --image      test_img.jpg \
    --device     rtx-5070ti \
    --out-dir    benchmarks/results

# Compare devices
python benchmarks/scripts/compare_devices.py benchmarks/results/
```

</details>

Benchmark methodology: 50-iteration warm-up, 500 measurement iterations. Wall time measured with `std::chrono::steady_clock`; GPU time with `cudaEventRecord`. Latency = end-to-end including preprocessing, inference, and postprocessing.

---

## Build Options

| CMake option              | Default | Description                             |
|---------------------------|---------|-----------------------------------------|
| `RFDETR_BUILD_APPS`       | ON      | CLI binaries                            |
| `RFDETR_BUILD_BENCHMARKS` | OFF     | Benchmark harness (`rfdetr_bench`)      |
| `RFDETR_BUILD_C_API`      | OFF     | C ABI shared library (`librfdetr_c`)    |
| `RFDETR_BUILD_SHARED`     | ON      | Shared library (vs static)              |
| `RFDETR_USE_OPENCV_CUDA`  | OFF     | Use OpenCV CUDA module for GPU ops      |

---

## INT8 Quantization

```sh
# 1. Generate calibration cache (Python, one-time)
python trt-files/scripts/int8_calibrate.py \
    --onnx   trt-files/onnx/rf-detr-nano.onnx \
    --images /path/to/calibration_images \
    --output trt-files/onnx/rf-detr-nano.calib \
    --max-images 500

# 2a. TRT < 11: C++ builder with cache
./build/rfdetr_build \
    --onnx      trt-files/onnx/rf-detr-nano.onnx \
    --precision int8 \
    --calib     trt-files/onnx/rf-detr-nano.calib

# 2b. Any TRT: trtexec wrapper
python trt-files/scripts/build_engine.py \
    --onnx       trt-files/onnx/rf-detr-nano.onnx \
    --precision  int8 \
    --int8-calib trt-files/onnx/rf-detr-nano.calib
```

> **TRT 11 note:** TRT 11 uses strongly-typed networks — implicit quantization flags are removed. Use `build_engine.py` (which calls `trtexec`) for INT8 on TRT 11+.

---

## Docker

```sh
# x86_64 dev environment
docker build -f docker/Dockerfile.x86_64 --target dev -t rfdetr-cpp:dev .
docker run --gpus all --rm -it \
    -v $(pwd)/trt-files:/workspace/trt-files \
    -v $(pwd)/out:/workspace/out \
    rfdetr-cpp:dev bash

# x86_64 production (binaries only, minimal image)
docker build -f docker/Dockerfile.x86_64 --target prod -t rfdetr-cpp:prod .

# Jetson (run on-device)
docker build -f docker/Dockerfile.jetson -t rfdetr-cpp:jetson .
docker run --runtime nvidia --rm -it \
    -v $(pwd)/trt-files:/workspace/trt-files \
    -v $(pwd)/out:/workspace/out \
    rfdetr-cpp:jetson bash

# Docker Compose
docker compose -f docker/docker-compose.yml run --rm x86_dev bash
```

See [docs/jetson.md](docs/jetson.md) for the full Jetson deployment guide including JetPack version matrix, clock pinning, and memory limits per Orin NX variant.

---

## Architecture

```
cv::Mat (BGR, host)
    │
    │  memcpy → pinned staging buffer
    ▼
Pinned Host ──cudaMemcpyAsync──► Device uint8 (raw BGR)
                                      │
                          ┌───────────┘
                          ▼
              squareResizeNormalizeKernel()       ← single CUDA kernel
              ┌─────────────────────────────┐
              │  square forced resize        │
              │  BGR → RGB                  │
              │  ImageNet normalize (÷255,   │
              │    subtract mean, div std)   │
              └────────────┬────────────────┘
                           ▼
              TRT input buffer (float32, device)
                           │
                  enqueueV3() / cudaGraphLaunch()
                           │
                           ▼
              TRT output buffers: dets · labels · [masks]
                           │
                  cudaMemcpyAsync D→H
                           │
                           ▼
              Sigmoid + top-K filter (no NMS)
              [+ bilinear upsample + threshold for seg]
                           │
                           ▼
                      Detections
```

### Key Design Differences vs YOLO

| | RF-DETR | YOLO |
|---|---|---|
| Architecture | DETR transformer + DINOv2 | CNN / ViT hybrid |
| Postprocess | Sigmoid + top-K, **no NMS** | Decode grid anchors + NMS |
| Input resize | **Square forced resize** | Letterbox (preserve aspect) |
| Segmentation | Per-query mask logits (Mask2Former-style) | Prototype masks + coefficients |
| Output layout | `[dets, labels, masks]` tensors | Concatenated anchor grid |

---

## Project Structure

```
rf-detr-cpp/
├── include/rfdetr/
│   ├── core/               # TrtSession, CUDA kernel, postprocess, drawing, types
│   ├── tasks/              # RFDetrDetector, RFDetrSegmenter
│   └── c_api.h             # Stable C ABI header
├── src/
│   ├── core/               # Implementation
│   └── tasks/              # Detector + segmenter implementation
├── apps/                   # CLI binaries
├── benchmarks/
│   ├── scripts/            # run_all.sh, compare_devices.py
│   └── results/            # JSON reports (git-ignored)
├── trt-files/
│   └── scripts/            # export_onnx.py, int8_calibrate.py, build_engine.py
├── docker/                 # Dockerfile.x86_64, Dockerfile.jetson, docker-compose.yml
├── docs/                   # jetson.md
└── cmake/                  # FindTensorRT.cmake
```

---

## Requirements

- CUDA 12.x
- TensorRT 10.x or 11.x
- OpenCV 4.5+
- CMake 3.20+
- C++17 compiler (GCC 9+, Clang 10+)

---

## License

Apache-2.0. See [LICENSE](./LICENSE).

---

## Acknowledgments

- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr) — model architecture and pretrained weights
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — optimized GPU inference
- [OpenCV](https://github.com/opencv/opencv) — image I/O and visualization
- [YOLOs-CPP-TensorRT](https://github.com/Geekgineer/YOLOs-CPP-TensorRT) — TensorRT C++ pipeline inspiration
