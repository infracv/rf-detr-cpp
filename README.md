<img width="1540" height="324" alt="RF-DETR CPP" src="https://github.com/user-attachments/assets/2b21c5d5-2ba7-498d-94f1-fe3f8e2bf871" />

<h3 align="center">Production-ready C++/TensorRT inference engine for RF-DETR.</h3>

<p align="center">
  <a href="#quick-start">Quick Start</a> ·
  <a href="#supported-models--tasks">Models</a> ·
  <a href="#api-reference">API</a> ·
  <a href="#benchmarks">Benchmarks</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/TensorRT-%E2%89%A5%2010.0-76B900?logo=nvidia&logoColor=white" alt="TensorRT"/>
  <img src="https://img.shields.io/badge/CUDA-%E2%89%A5%2012.0-76B900?logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg?logo=cplusplus&logoColor=white" alt="C++17"/>
  <img src="https://img.shields.io/badge/license-Apache--2.0-ef4444" alt="License"/>
  <a href="https://github.com/infracv/rf-detr-cpp/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/infracv/rf-detr-cpp.svg"></a>
</p>


---

## Overview

**RF-DETR C++** is a production-grade TensorRT inference engine for [RF-DETR](https://github.com/roboflow/rf-detr), Roboflow's transformer-based real-time object detection model built on a DINOv2 backbone.

```cpp
#include "rfdetr/tasks/detector.hpp"

rfdetr::RFDetrDetector detector("rf-detr-nano-fp32.engine");
rfdetr::Detections dets = detector.detect(cv::imread("image.jpg"), 0.5f);
```

Unlike YOLO models, RF-DETR uses a DETR-style architecture with **no NMS, no anchor grids, no letterboxing**. This requires a different inference pipeline, which this library implements entirely in C++ with zero Python at runtime.

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

| Task | Variants |
|:-----|:---------|
| Object Detection | `nano`, `small`, `medium`, `base`, `large` |
| Instance Segmentation | `seg-nano`, `seg-small`, `seg-medium`, `seg-large`, `seg-xlarge`, `seg-2xlarge` |

### Precision support

| Precision | TRT < 11 | TRT 11+ | Notes |
|:---------:|:--------:|:-------:|-------|
| FP32 | ✅ | ✅ | Default |
| FP16 | ✅ (`kFP16` flag) | ✅ (`convert_fp16.py` → pre-converted ONNX) | ~25% faster than FP32 |
| INT8 | ✅ (calibration cache) | ✅ (QDQ ONNX via `convert_int8.py`) | Lowest memory |

> **FP16 NaN sanity check.** RF-DETR's transformer attention can produce values that exceed the FP16 range (±65,504), causing NaNs to propagate through the network and yield zero detections. After building an FP16 engine, run a single detection on `asset/test_img.jpg`. If you get zero detections on an image where FP32 finds objects, your FP16 engine is hitting overflow. Fall back to FP32 (`rfdetr_build --precision fp32`) or rebuild with mixed-precision settings.

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
| Jetson Orin | `87` |
| Thor / GH200 | `101` |

The default build (no `-DCMAKE_CUDA_ARCHITECTURES`) compiles for all five (86 87 89 101 120). Pass a single value to keep build times short.

> **Note:** Always pass `-DCMAKE_CUDA_ARCHITECTURES` explicitly. If your environment has a stale value set (e.g. from a conda env), CMake will use that instead and the CUDA preprocessing kernel won't be optimized for your GPU.

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

For INT8 see [trt-files/INT8_QUANTIZATION.md](./trt-files/INT8_QUANTIZATION.md).

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

// Check whether CUDA Graph was captured at construction:
if (detector.cuda_graph_active())
    std::puts("CUDA Graph active: lowest dispatch latency");
```

### Instance Segmentation

```cpp
#include "rfdetr/tasks/segmenter.hpp"

rfdetr::RFDetrSegmenter segmenter("rf-detr-seg-nano-fp16.engine");

cv::Mat image = cv::imread("image.jpg");
rfdetr::Detections dets = segmenter.segment(image, /*threshold=*/0.5f);

for (const auto& d : dets)
    std::printf("class=%d  score=%.2f  mask=%dx%d (CV_8UC1)\n",
                d.class_id, d.score, d.mask.cols, d.mask.rows);

rfdetr::draw_segmentations(image, dets);
rfdetr::draw_detections(image, dets);
cv::imwrite("out.jpg", image);
```

Each `Detection.mask` is a `CV_8UC1` cv::Mat of the original image size, with values 0 or 255.

### Batch Inference

```cpp
std::vector<cv::Mat> frames = { img0, img1, img2, img3 };

rfdetr::RFDetrDetector detector("rf-detr-nano-fp32.engine");
auto batch = detector.detect_batch(frames, 0.5f);

for (std::size_t i = 0; i < batch.size(); ++i)
    std::printf("frame %zu: %zu detections\n", i, batch[i].size());
```

### C ABI (Python / Rust / Go FFI)

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

## Benchmarks

NVIDIA RTX 5070 Ti, 500 iters, 50-iter warm-up, Batch 1.

| Task | Precision | FPS | Avg Latency | P50 | P99 | GPU Memory |
|:-----|:---------:|:---:|:-----------:|:---:|:---:|:----------:|
| Detection (`nano`) | **FP32** | 406 | 2.462 ms | 2.444 ms | 2.778 ms | 859 MB |
| Detection (`nano`) | **FP16** | 514 | 1.944 ms | 1.911 ms | 2.508 ms | 836 MB |
| Segmentation (`seg-nano`) | **FP32** | 112 | 8.906 ms | 8.817 ms | 10.243 ms | 892 MB |
| Segmentation (`seg-nano`) | **FP16** | 157 | 6.364 ms | 6.314 ms | 7.211 ms | 886 MB |

> Numbers include the full pipeline: preprocessing, inference, and postprocessing. Segmentation mask decoding runs entirely on the GPU via a dedicated CUDA kernel.

<details>
<summary>Jetson Orin NX 16GB — TensorRT 10.3.0 / CUDA 12.6</summary>

500 iters, 50-iter warm-up, Batch 1.

| Task | Precision | FPS | Avg Latency | P50 | P99 |
|:-----|:---------:|:---:|:-----------:|:---:|:---:|
| Detection (`nano`) | **FP16** | 120 | 8.302 ms | 7.766 ms | 12.829 ms |
| Detection (`nano`) | **FP32** | 50 | 19.984 ms | 18.858 ms | 33.003 ms |
| Segmentation (`seg-nano`) | **FP16** | 55 | 18.093 ms | 17.008 ms | 23.871 ms |
| Segmentation (`seg-nano`) | **FP32** | 24 | 42.292 ms | 42.047 ms | 45.436 ms |

> GPU Memory column is omitted — Jetson uses unified CPU/GPU memory, so `cudaMemGetInfo` reports system memory rather than dedicated VRAM and the delta reads as zero. See the [Benchmarking Guide](./benchmarks/BENCHMARKING.md#unified-memory-caveat) for details.

</details>

See [benchmarks/BENCHMARKING.md](./benchmarks/BENCHMARKING.md) to reproduce these numbers or run your own benchmarks on any GPU.

---

## License

Apache-2.0. See [LICENSE](./LICENSE).

---

## Acknowledgments

- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr) for the model architecture and pretrained weights
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) for optimised GPU inference
- [OpenCV](https://github.com/opencv/opencv) for image I/O and visualisation

---

## Contributing

We welcome and appreciate all contributions. If you notice any issues or bugs, have questions, or would like to suggest new features, please open an issue or pull request. By sharing your ideas and improvements, you help make RF-DETR C++ better for everyone.

See [CONTRIBUTING.md](./.github/CONTRIBUTING.md) for guidelines on setting up the dev environment, coding standards, and the PR process.

---

<p align="center">
  ⭐ If RF-DETR C++ is useful to you, consider giving it a star on <a href="https://github.com/infracv/rf-detr-cpp">GitHub</a>. It helps others find the project.
</p>

