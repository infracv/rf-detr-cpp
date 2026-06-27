<img width="1540" height="324" alt="RF-DETR CPP" src="https://github.com/user-attachments/assets/2b21c5d5-2ba7-498d-94f1-fe3f8e2bf871" />

<h3 align="center">Production-Ready RF-DETR Inference Engine for C++</h3>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#supported-models--tasks">Models</a> ·
  <a href="#api-reference">API</a> ·
  <a href="#benchmarks">Benchmarks</a>
</p>

<p align="center">
  <a href="https://github.com/infracv/rf-detr-cpp/stargazers"><img src="https://img.shields.io/github/stars/infracv/rf-detr-cpp?style=flat-square&logo=github&color=f5c211" alt="Stars"/></a>
  <a href="https://github.com/infracv/rf-detr-cpp/network/members"><img src="https://img.shields.io/github/forks/infracv/rf-detr-cpp?style=flat-square&logo=github&color=3b82f6" alt="Forks"/></a>
  <img src="https://img.shields.io/badge/TensorRT-%E2%89%A5%2010.0-76B900?logo=nvidia&logoColor=white" alt="TensorRT"/>
  <img src="https://img.shields.io/badge/CUDA-%E2%89%A5%2012.0-76B900?logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg?logo=cplusplus&logoColor=white" alt="C++17"/>
  <img src="https://img.shields.io/badge/license-Apache--2.0-ef4444" alt="License"/>
</p>


---

## Latest News

* **[2026.07.02]** Initial release v0.0.1 with object detection support.

---

## What is RF-DETR C++?

**RF-DETR C++** is a production-grade TensorRT inference engine for [RF-DETR](https://github.com/roboflow/rf-detr) — Roboflow's transformer-based real-time object detection model built on a DINOv2 backbone.

```cpp
#include "rfdetr/tasks/detector.hpp"

rfdetr::RFDetrDetector detector("rf-detr-nano-fp32.engine");
rfdetr::Detections dets = detector.detect(cv::imread("image.jpg"), 0.5f);
```

Unlike YOLO models, RF-DETR uses a DETR-style architecture — **no NMS, no anchor grids, no letterboxing**. This requires a different inference pipeline, which this library implements entirely in C++ with zero Python at runtime.

### Why RF-DETR C++?

Most RF-DETR deployments run Python at inference time. This library eliminates that dependency:

| | RF-DETR C++ | Python (PyTorch) |
|---|:---:|:---:|
| Runtime dependency | **C++ only** | Python + PyTorch + CUDA |
| Preprocessing | **GPU CUDA kernel** | CPU / torchvision |
| Host-to-device transfer | **Async** (pinned memory) | Synchronous |
| Precision | **FP32 / FP16 / INT8** | FP32 / FP16 |
| Inference dispatch | **CUDA Graph replay** | Per-frame forward pass |
| Latency (RTX 5070 Ti) | **2.0 ms (FP16)** | ~15–30 ms |

---

## Quick Start

**3 steps to your first inference:**

```sh
# 1. Clone & build
git clone https://github.com/infracv/rf-detr-cpp.git
cd rf-detr-cpp
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)
# Replace 120 with your GPU arch: RTX 30xx=86, RTX 40xx=89, RTX 50xx=120
# Tarball TensorRT: add -DTENSORRT_DIR=/path/to/TensorRT

# 2. Export ONNX and build TRT engine (one-time, Python)
pip install -r requirements.txt
python trt-files/scripts/export_onnx.py --variant nano --out-dir trt-files/onnx

# FP32 engine
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp32

# FP16 engine
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp16

# 3. Run inference
./build/rfdetr_detect \
    --engine trt-files/onnx/rf-detr-nano-fp16.engine \
    --image  asset/test_img.jpg --out out/result.jpg
```

---

## Supported Models & Tasks

| Variant | Task | Input | COCO mAP |
|---------|:----:|:-----:|:--------:|
| `nano` | Detection | 384×384 | ~43 |
| `small` | Detection | 512×512 | ~48 |
| `medium` | Detection | 576×576 | ~51 |
| `base` | Detection | 560×560 | ~53 |
| `large` | Detection | 704×704 | ~55 |

### Precision support

| Precision | TRT < 11 | TRT 11+ | Notes |
|:---------:|:--------:|:-------:|-------|
| FP32 | ✅ | ✅ | Default |
| FP16 | ✅ (`kFP16` flag) | ✅ (`convert_fp16.py` → pre-converted ONNX) | ~25% faster than FP32 |
| INT8 | ✅ (calibration cache) | ✅ (QDQ ONNX via `convert_int8.py`) | Lowest memory |

---

## Installation

### Prerequisites

| Dependency | Version | Notes |
|:-----------|:-------:|:------|
| NVIDIA GPU | CC ≥ 8.0 | RTX 30xx / 40xx / 50xx |
| CUDA Toolkit | ≥ 12.0 | |
| TensorRT | ≥ 10.0 | TRT 11 supported (strongly-typed networks) |
| OpenCV | ≥ 4.5 | core, imgproc, imgcodecs, videoio, highgui |
| CMake | ≥ 3.20 | |
| C++ compiler | C++17 | GCC 9+, Clang 10+ |

### Build from Source

```sh
git clone https://github.com/infracv/rf-detr-cpp.git
cd rf-detr-cpp
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)
# Tarball TensorRT: add -DTENSORRT_DIR=/path/to/TensorRT
```

Replace `120` with your GPU's compute capability:

| GPU family | `CMAKE_CUDA_ARCHITECTURES` |
|:-----------|:--------------------------:|
| RTX 30xx (Ampere) | `86` |
| RTX 40xx (Ada Lovelace) | `89` |
| RTX 50xx (Blackwell) | `120` |

> **Note:** Always pass `-DCMAKE_CUDA_ARCHITECTURES` explicitly. If your environment has a stale value set (e.g. from a conda env), CMake will use that instead and the CUDA preprocessing kernel won't be optimized for your GPU.

### Install TensorRT

<details>
<summary><b>Ubuntu (apt)</b></summary>

```sh
sudo apt update && sudo apt install -y tensorrt
```

</details>

<details>
<summary><b>Manual / tarball</b></summary>

Download from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt), extract, and pass `-DTENSORRT_DIR=/path/to/TensorRT` to CMake.

</details>

---

## Model Conversion

Python is only needed for this one-time conversion step. The C++ runtime requires no Python.

```sh
# Python deps (uv recommended)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Export RF-DETR PyTorch → ONNX
python trt-files/scripts/export_onnx.py --variant nano --out-dir trt-files/onnx

# FP32 engine
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp32

# FP16 engine
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp16
```

For INT8 see the [INT8 Quantization](#int8-quantization) section.

---

## API Reference

### Object Detection

```cpp
#include "rfdetr/tasks/detector.hpp"

rfdetr::RFDetrDetector detector("rf-detr-nano-fp32.engine");

cv::Mat image = cv::imread("image.jpg");
rfdetr::Detections dets = detector.detect(image, /*threshold=*/0.5f);

for (const auto& d : dets)
    std::printf("class=%d  score=%.2f  box=[%.1f %.1f %.1f %.1f]\n",
                d.class_id, d.score, d.box.x1, d.box.y1, d.box.x2, d.box.y2);

rfdetr::draw_detections(image, dets);
cv::imwrite("out.jpg", image);
```

### Batch Inference

```cpp
std::vector<cv::Mat> frames = { img0, img1, img2, img3 };

rfdetr::RFDetrDetector detector("rf-detr-nano-fp32.engine");
auto batch = detector.detect_batch(frames, 0.5f);

for (std::size_t i = 0; i < batch.size(); ++i)
    std::printf("frame %zu: %zu detections\n", i, batch[i].size());
```

### C ABI — Python / Rust / Go FFI

Build with `-DRFDETR_BUILD_C_API=ON` to produce `librfdetr_c.so`:

```c
#include "rfdetr/c_api.h"

rfdetr_detector_t* det = rfdetr_detector_create("model.engine", NULL);
rfdetr_detections_t* res = rfdetr_detector_detect(det, bgr_data, w, h, w*3, 0.5f);

for (int i = 0; i < res->count; ++i)
    printf("cls=%d  score=%.2f\n", res->detections[i].class_id, res->detections[i].score);

rfdetr_detections_free(res);
rfdetr_detector_destroy(det);
```

```python
# Python via ctypes
import ctypes
lib = ctypes.CDLL("librfdetr_c.so")
# See include/rfdetr/c_api.h for the full binding
```

---

## CLI Tools

| Binary | Description |
|--------|-------------|
| `rfdetr_build` | Convert ONNX → TensorRT engine (FP32 / FP16 / INT8) |
| `rfdetr_detect` | Detection on a single image |
| `rfdetr_video` | Detection on a video file or camera stream |
| `rfdetr_batch` | Multi-image batch inference |
| `rfdetr_bench` | Latency + throughput benchmark (image / video / camera modes) |
| `rfdetr_inspect` | Print engine I/O bindings and shapes |
| `rfdetr_smoke` | Quick forward-pass sanity check |

```sh
# Detect on image
./build/rfdetr_detect --engine rf-detr-nano-fp16.engine --image test_img.jpg --out out.jpg

# Run on video
./build/rfdetr_video --engine rf-detr-nano-fp16.engine --input video.mp4 --out out.mp4
```

---

## Benchmarks

### NVIDIA RTX 5070 Ti — RF-DETR nano · 384×384 · 500 iters · 50-iter warm-up · Batch 1

| Precision | FPS | Avg Latency | P50 | P99 | GPU Memory |
|:---------:|:---:|:-----------:|:---:|:---:|:----------:|
| **FP32** | **401** | 2.496 ms | 2.479 ms | 2.838 ms | 831 MB |
| **FP16** | **500** | 2.001 ms | 1.972 ms | 2.703 ms | 768 MB |
| **INT8** | **482** | 2.073 ms | 2.045 ms | 2.786 ms | 684 MB |

> Numbers include the full pipeline — preprocessing, inference, and postprocessing. Wall time measured with `std::chrono::steady_clock`; GPU time with `cudaEventRecord`.

<details>
<summary><b>Run your own benchmarks</b></summary>

```sh
cmake -B build -S . -DRFDETR_BUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)

# Quick run (10 warm-up, 100 iters)
./build/rfdetr_bench quick rf-detr-nano-fp32.engine test_img.jpg

# Full image benchmark
./build/rfdetr_bench image rf-detr-nano-fp32.engine test_img.jpg \
    --device rtx-5070ti --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results

# Video benchmark
./build/rfdetr_bench video rf-detr-nano-fp32.engine test_video.mp4 \
    --device rtx-5070ti --iters 500

# Full sweep over all *.engine files
./benchmarks/scripts/run_all.sh \
    --engine-dir trt-files/onnx --image test_img.jpg \
    --device rtx-5070ti --out-dir benchmarks/results

# Compare across devices
python benchmarks/scripts/compare_devices.py benchmarks/results/
```

</details>

---

## Build Options

| CMake option | Default | Description |
|---|:---:|---|
| `RFDETR_BUILD_APPS` | ON | CLI binaries |
| `RFDETR_BUILD_EXAMPLES` | OFF | Example programs under `examples/` |
| `RFDETR_BUILD_BENCHMARKS` | OFF | Benchmark harness (`rfdetr_bench`) |
| `RFDETR_BUILD_C_API` | OFF | C ABI shared library (`librfdetr_c`) |
| `RFDETR_BUILD_SHARED` | ON | Shared library (vs static) |
| `RFDETR_USE_OPENCV_CUDA` | OFF | Use OpenCV CUDA module for GPU ops |

---

## INT8 Quantization

Two paths depending on TensorRT version.

### Calibration dataset — COCO val2017 (recommended)

RF-DETR is trained on COCO 80 classes. NVIDIA recommends ~5,000 representative images for COCO-trained models. **Labels are not used** — only pixel values matter.

```sh
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip   # → val2017/ with 5,000 images
```

| Calibration images | Quality | Use case |
|:------------------:|---------|----------|
| 100–200 | Quick smoke test | CI / fast iteration |
| 1,000–2,000 | Good | Prototype |
| **5,000 (COCO val2017)** | **Best** | **Production** |

### TRT 11+ — QDQ path

```sh
# Step 1: Insert Quantize/Dequantize nodes via calibration
python trt-files/scripts/convert_int8.py \
    --onnx trt-files/onnx/rf-detr-nano.onnx \
    --images val2017/ --max-images 5000 \
    --out trt-files/onnx/rf-detr-nano-int8.onnx

# Step 2: Build engine — TRT reads the embedded QDQ nodes
./build/rfdetr_build \
    --onnx   trt-files/onnx/rf-detr-nano-int8.onnx \
    --engine trt-files/onnx/rf-detr-nano-int8.engine \
    --precision fp32
```

### TRT < 11 — Implicit calibration path

```sh
# Step 1: Generate calibration cache
python trt-files/scripts/int8_calibrate.py \
    --onnx trt-files/onnx/rf-detr-nano.onnx \
    --images val2017/ --max-images 5000 \
    --output trt-files/onnx/rf-detr-nano.calib

# Step 2: Build with C++ builder
./build/rfdetr_build \
    --onnx trt-files/onnx/rf-detr-nano.onnx --precision int8 \
    --calib trt-files/onnx/rf-detr-nano.calib
```

**Tips:**
- `convert_int8.py` excludes `LayerNorm`, `Softmax`, and `Gelu` by default — pass `--no-exclude-sensitive` to quantize everything.
- QDQ models and `.calib` caches are GPU-architecture-specific; rebuild when changing hardware.
- The `val2017/` folder can be reused across all model variants.

---

## License

Apache-2.0. See [LICENSE](./LICENSE).

---

## Acknowledgments

- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr) — model architecture and pretrained weights
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — optimised GPU inference
- [OpenCV](https://github.com/opencv/opencv) — image I/O and visualisation
- [YOLOs-CPP-TensorRT](https://github.com/Geekgineer/YOLOs-CPP-TensorRT) — TensorRT C++ pipeline inspiration

---

<p align="center">
  <strong>⭐ If RF-DETR C++ helps your project, consider giving it a star!</strong>
</p>
