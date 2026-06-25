<h1 align="center">RF-DETR C++</h1>
<h3 align="center">Production-Ready RF-DETR Inference Engine for C++ / TensorRT</h3>

<p align="center">
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-supported-models--tasks">Models</a> ·
  <a href="#-api-reference">API</a> ·
  <a href="#-benchmarks">Benchmarks</a> ·
  <a href="#-docker">Docker</a> ·
  <a href="docs/jetson.md">Jetson Guide</a>
</p>

<p align="center">
  <a href="https://github.com/infracv/rf-detr-cpp/stargazers"><img src="https://img.shields.io/github/stars/infracv/rf-detr-cpp?style=flat-square&logo=github&color=f5c211" alt="Stars"/></a>
  <a href="https://github.com/infracv/rf-detr-cpp/network/members"><img src="https://img.shields.io/github/forks/infracv/rf-detr-cpp?style=flat-square&logo=github&color=3b82f6" alt="Forks"/></a>
  <img src="https://img.shields.io/badge/TensorRT-%E2%89%A5%2010.0-76B900?logo=nvidia&logoColor=white" alt="TensorRT"/>
  <img src="https://img.shields.io/badge/CUDA-%E2%89%A5%2012.0-76B900?logo=nvidia&logoColor=white" alt="CUDA"/>
  <img src="https://img.shields.io/badge/C%2B%2B-17-blue.svg?logo=cplusplus&logoColor=white" alt="C++17"/>
  <img src="https://img.shields.io/badge/license-Apache--2.0-ef4444" alt="License"/>
</p>

<p align="center">
  <img src="asset/RF-DETR CPP.png" alt="RF-DETR C++" width="100%"/>
</p>

---

## 📰 Latest News

* **[2026.07.02]** Initial release v0.0.1 with object detection support.

---

## What is RF-DETR C++?

**RF-DETR C++** is a production-grade TensorRT inference engine for [RF-DETR](https://github.com/roboflow/rf-detr) — Roboflow's transformer-based real-time object detection and segmentation model built on a DINOv2 backbone.

```cpp
#include "rfdetr/tasks/detector.hpp"

rfdetr::RFDetrDetector detector("rf-detr-nano-fp32.engine");
rfdetr::Detections dets = detector.detect(cv::imread("image.jpg"), 0.5f);
```

Unlike YOLO models, RF-DETR uses a DETR-style architecture — **no NMS, no anchor grids, no letterboxing**. This requires a different inference pipeline, which this library implements entirely in C++ with zero Python at runtime.

### Why RF-DETR C++?

Most RF-DETR deployments run Python at inference time. This library eliminates that dependency:

| | RF-DETR C++ | Python inference |
|---|:---:|:---:|
| Runtime dependency | **C++ only** | Python + PyTorch |
| Preprocessing | **GPU CUDA kernel** | CPU / torchvision |
| Precision | **FP32 / FP16 / INT8** | FP32 only |
| Deployment | **Jetson, Docker, bare-metal** | x86 workstation |
| Latency (RTX 5070 Ti) | **2.0 ms (FP16)** | ~15–30 ms |

### Architecture differences vs YOLO

| | RF-DETR | YOLO |
|---|---|---|
| Backbone | DINOv2 transformer | CNN / ViT hybrid |
| Postprocess | Sigmoid + top-K, **no NMS** | Decode anchor grid + NMS |
| Input resize | **Square forced resize** | Letterbox (preserve aspect) |
| Segmentation | Per-query mask logits (Mask2Former-style) | Prototype masks + coefficients |
| Output tensors | `dets · labels · [masks]` | Single concatenated tensor |

---

## ⚡ Quick Start

**3 steps to your first inference:**

```sh
# 1. Clone & build
git clone https://github.com/infracv/rf-detr-cpp.git
cd rf-detr-cpp
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DTENSORRT_DIR=/path/to/TensorRT
cmake --build build -j$(nproc)

# 2. Export ONNX and build TRT engine (one-time, Python)
pip install -r requirements.txt
python trt-files/scripts/export_onnx.py --variant nano --out-dir trt-files/onnx
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp16

# 3. Run inference
./build/rfdetr_detect \
    --engine trt-files/onnx/rf-detr-nano-fp16.engine \
    --image  test_img.jpg --out out/result.jpg
```

**In your own project — one include:**

```cpp
#include "rfdetr/tasks/detector.hpp"

rfdetr::RFDetrDetector det("rf-detr-nano-fp16.engine");
auto results = det.detect(frame, 0.5f);
for (auto& d : results)
    std::printf("cls=%d  score=%.2f  [%.0f %.0f %.0f %.0f]\n",
                d.class_id, d.score, d.box.x1, d.box.y1, d.box.x2, d.box.y2);
```

---

## 🧩 Supported Models & Tasks

| Variant | Task | Input | COCO mAP |
|---------|:----:|:-----:|:--------:|
| `nano` | Detection | 384×384 | ~43 |
| `small` | Detection | 512×512 | ~48 |
| `medium` | Detection | 576×576 | ~51 |
| `base` | Detection | 560×560 | ~53 |
| `large` | Detection | 704×704 | ~55 |
| `seg-nano` | Detection + Segmentation | 312×312 | — |
| `seg-medium` | Detection + Segmentation | 432×432 | — |
| `seg-large` | Detection + Segmentation | 504×504 | — |

### Precision support

| Precision | TRT < 11 | TRT 11+ | Notes |
|:---------:|:--------:|:-------:|-------|
| FP32 | ✅ | ✅ | Default |
| FP16 | ✅ (`kFP16` flag) | ✅ (`convert_fp16.py` → pre-converted ONNX) | ~25% faster than FP32 |
| INT8 | ✅ (calibration cache) | ✅ (QDQ ONNX via `convert_int8.py`) | Lowest memory |

---

## 📦 Installation

### Prerequisites

| Dependency | Version | Notes |
|:-----------|:-------:|:------|
| NVIDIA GPU | CC ≥ 8.0 | RTX 30xx / 40xx / 50xx · Jetson Orin / AGX Thor |
| CUDA Toolkit | ≥ 12.0 | |
| TensorRT | ≥ 10.0 | TRT 11 supported (strongly-typed networks) |
| OpenCV | ≥ 4.5 | core, imgproc, imgcodecs, videoio, highgui |
| CMake | ≥ 3.20 | |
| C++ compiler | C++17 | GCC 9+, Clang 10+ |

### Build from Source

```sh
git clone https://github.com/infracv/rf-detr-cpp.git
cd rf-detr-cpp
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DTENSORRT_DIR=/path/to/TensorRT
cmake --build build -j$(nproc)
```

<details>
<summary><b>CUDA architecture flags (Jetson / custom GPU)</b></summary>

```sh
# RTX 50xx (Blackwell)
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=120

# Jetson Orin NX
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=87

# Jetson AGX Thor
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=101

# Multi-arch (one binary for all targets)
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES="87;101;120"
```

</details>

### Install TensorRT

<details>
<summary><b>Ubuntu (apt)</b></summary>

```sh
sudo apt update && sudo apt install -y tensorrt
```

</details>

<details>
<summary><b>Jetson (JetPack 6.x)</b></summary>

TensorRT is pre-installed. Verify:
```sh
dpkg -l | grep nvinfer
```

</details>

<details>
<summary><b>Manual / tarball</b></summary>

Download from [developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt), extract, and pass `-DTENSORRT_DIR=/path/to/TensorRT` to CMake.

</details>

---

## 🔄 Model Conversion

Python is only needed for this one-time conversion step. The C++ runtime requires no Python.

```sh
# Python deps (uv recommended)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Export RF-DETR PyTorch → ONNX
python trt-files/scripts/export_onnx.py --variant nano --out-dir trt-files/onnx

# FP32 engine (baseline)
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp32

# FP16 — TRT < 11
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp16

# FP16 — TRT 11+ (convert ONNX weights first, then build normally)
python trt-files/scripts/convert_fp16.py \
    --onnx trt-files/onnx/rf-detr-nano.onnx \
    --out  trt-files/onnx/rf-detr-nano-fp16.onnx
./build/rfdetr_build \
    --onnx trt-files/onnx/rf-detr-nano-fp16.onnx --precision fp32
```

<details>
<summary><b>INT8 quantization (COCO val2017 calibration)</b></summary>

```sh
# Download calibration dataset — one-time, 780 MB
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip

# TRT 11+: insert QDQ nodes, then build
python trt-files/scripts/convert_int8.py \
    --onnx trt-files/onnx/rf-detr-nano.onnx --images val2017/ \
    --out  trt-files/onnx/rf-detr-nano-int8.onnx --max-images 5000
./build/rfdetr_build \
    --onnx trt-files/onnx/rf-detr-nano-int8.onnx --precision fp32

# TRT < 11: generate calibration cache, then build
python trt-files/scripts/int8_calibrate.py \
    --onnx trt-files/onnx/rf-detr-nano.onnx --images val2017/ \
    --output trt-files/onnx/rf-detr-nano.calib --max-images 5000
./build/rfdetr_build \
    --onnx trt-files/onnx/rf-detr-nano.onnx --precision int8 \
    --calib trt-files/onnx/rf-detr-nano.calib
```

See the [INT8 Quantization](#-int8-quantization) section for full details.

</details>

> **Note:** TensorRT engines are GPU-architecture and TRT-version specific. Always rebuild on the target hardware.

---

## 📖 API Reference

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

### Instance Segmentation

```cpp
#include "rfdetr/tasks/segmenter.hpp"

rfdetr::RFDetrSegmenter seg("rf-detr-seg-nano-fp32.engine");

rfdetr::Detections dets = seg.segment(image, /*threshold=*/0.5f);
// Each Detection has a `.mask` field: CV_8UC1, same size as input, values 0 or 255

rfdetr::draw_segmentations(image, dets);
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

## 🛠️ CLI Tools

| Binary | Description |
|--------|-------------|
| `rfdetr_build` | Convert ONNX → TensorRT engine (FP32 / FP16 / INT8) |
| `rfdetr_detect` | Detection on a single image |
| `rfdetr_video` | Detection on a video file or camera stream |
| `rfdetr_seg` | Segmentation on image or video |
| `rfdetr_batch` | Multi-image batch inference |
| `rfdetr_bench` | Latency + throughput benchmark (image / video / camera modes) |
| `rfdetr_inspect` | Print engine I/O bindings and shapes |
| `rfdetr_smoke` | Quick forward-pass sanity check |

```sh
# Detect on image
./build/rfdetr_detect --engine rf-detr-nano-fp16.engine --image test_img.jpg --out out.jpg

# Run on video
./build/rfdetr_video --engine rf-detr-nano-fp16.engine --input video.mp4 --out out.mp4

# Segmentation
./build/rfdetr_seg --engine rf-detr-seg-nano-fp32.engine --image test_img.jpg --out out.jpg
```

---

## 📊 Benchmarks

### NVIDIA RTX 5070 Ti — RF-DETR nano · 384×384 · 500 iters · 50-iter warm-up · Batch 1

| Precision | FPS | Avg Latency | P50 | P99 | GPU Memory |
|:---------:|:---:|:-----------:|:---:|:---:|:----------:|
| **FP32** | **401** | 2.496 ms | 2.479 ms | 2.838 ms | 831 MB |
| **FP16** | **500** | 2.001 ms | 1.972 ms | 2.703 ms | 768 MB |
| **INT8** | **482** | 2.073 ms | 2.045 ms | 2.786 ms | 684 MB |

### NVIDIA Jetson AGX Thor — RF-DETR nano · 384×384 · 500 iters · 50-iter warm-up · Batch 1

| Precision | FPS | Avg Latency | P50 | P99 | GPU Memory |
|:---------:|:---:|:-----------:|:---:|:---:|:----------:|
| **FP32** | — | — ms | — ms | — ms | — MB |
| **FP16** | — | — ms | — ms | — ms | — MB |
| **INT8** | — | — ms | — ms | — ms | — MB |

### NVIDIA Jetson Orin NX 16 GB — RF-DETR nano · 384×384 · 500 iters · 50-iter warm-up · Batch 1

| Precision | FPS | Avg Latency | P50 | P99 | GPU Memory |
|:---------:|:---:|:-----------:|:---:|:---:|:----------:|
| **FP32** | — | — ms | — ms | — ms | — MB |
| **FP16** | — | — ms | — ms | — ms | — MB |
| **INT8** | — | — ms | — ms | — ms | — MB |

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

## ⚙️ Build Options

| CMake option | Default | Description |
|---|:---:|---|
| `RFDETR_BUILD_APPS` | ON | CLI binaries |
| `RFDETR_BUILD_BENCHMARKS` | OFF | Benchmark harness (`rfdetr_bench`) |
| `RFDETR_BUILD_C_API` | OFF | C ABI shared library (`librfdetr_c`) |
| `RFDETR_BUILD_SHARED` | ON | Shared library (vs static) |
| `RFDETR_USE_OPENCV_CUDA` | OFF | Use OpenCV CUDA module for GPU ops |

---

## 🔢 INT8 Quantization

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

## 🐳 Docker

```sh
# x86_64 dev environment
docker build -f docker/Dockerfile.x86_64 --target dev -t rfdetr-cpp:dev .
docker run --gpus all --rm -it \
    -v $(pwd)/trt-files:/workspace/trt-files \
    -v $(pwd)/out:/workspace/out \
    rfdetr-cpp:dev bash

# x86_64 production (minimal image, binaries only)
docker build -f docker/Dockerfile.x86_64 --target prod -t rfdetr-cpp:prod .

# Jetson (run on-device)
docker build -f docker/Dockerfile.jetson -t rfdetr-cpp:jetson .
docker run --runtime nvidia --rm -it \
    -v $(pwd)/trt-files:/workspace/trt-files \
    rfdetr-cpp:jetson bash

# Docker Compose
docker compose -f docker/docker-compose.yml run --rm x86_dev bash
```

See [docs/jetson.md](docs/jetson.md) for the full Jetson deployment guide including JetPack version matrix, clock pinning, and memory limits per Orin NX variant.

---

## 🏗️ Architecture

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
              │  ImageNet normalize          │
              │  (÷255, subtract mean/std)   │
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
              Sigmoid + top-K (no NMS)
              [+ bilinear upsample + threshold for seg]
                           │
                           ▼
                      Detections
```

### Performance optimisations

| Technique | Effect |
|---|---|
| **GPU square-resize kernel** | Single CUDA kernel: resize + BGR→RGB + normalize — no CPU preprocessing on the hot path |
| **Pinned host memory** | `cudaMallocHost` buffers enable truly async H2D transfers that overlap with compute |
| **CUDA Graph capture** | `enqueueV3` call graph captured once, replayed via `cudaGraphLaunch` — cuts ~0.1–0.3 ms dispatch overhead |
| **Single CUDA stream** | Minimal synchronisation — one `cudaStream_t` drives the full preprocess → infer → postprocess pipeline |
| **No NMS** | RF-DETR's set-prediction head skips anchor decoding and NMS entirely |

---

## 📂 Project Structure

```
rf-detr-cpp/
├── include/rfdetr/
│   ├── core/               # TrtSession, CUDA kernel, postprocess, drawing, types
│   ├── tasks/              # RFDetrDetector, RFDetrSegmenter
│   └── c_api.h             # Stable C ABI header
├── src/
│   ├── core/               # TrtSession, engine meta, postprocess, drawing
│   └── tasks/              # Detector + segmenter implementation
├── apps/                   # CLI binaries (detect, video, seg, bench, build, inspect …)
├── benchmarks/
│   ├── scripts/            # run_all.sh, compare_devices.py
│   └── results/            # CSV / JSON benchmark results
├── trt-files/
│   └── scripts/            # export_onnx.py, convert_fp16.py, convert_int8.py,
│                           #   int8_calibrate.py, build_engine.py, verify_engine.py
├── docker/                 # Dockerfile.x86_64, Dockerfile.jetson, docker-compose.yml
├── docs/                   # jetson.md
└── cmake/                  # FindTensorRT.cmake
```

---

## 🔧 Troubleshooting

<details>
<summary><b>TensorRT not found during CMake</b></summary>

```sh
# Check installation
dpkg -l | grep nvinfer

# Point to a custom location
cmake -B build -S . -DTENSORRT_DIR=/opt/TensorRT-10.x
```

</details>

<details>
<summary><b>CUDA Graph capture failed</b></summary>

Expected for transformer networks with data-dependent internal shapes (cross-attention). The library automatically falls back to `enqueueV3` — performance impact is minimal (~0.1–0.3 ms per frame).

</details>

<details>
<summary><b>Engine fails to load on a different machine</b></summary>

TensorRT engines are **GPU-architecture and TRT-version specific**. Always rebuild on the target device:

```sh
./build/rfdetr_build --onnx rf-detr-nano.onnx --precision fp16
```

</details>

<details>
<summary><b>FP16 build error on TRT 11+</b></summary>

TRT 11 uses strongly-typed networks — the `kFP16` builder flag is removed. Use `convert_fp16.py` to pre-convert the ONNX, then build with `--precision fp32`.

</details>

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes and open a Pull Request

---

## 📄 License

Apache-2.0. See [LICENSE](./LICENSE).

---

## 🙏 Acknowledgments

- [Roboflow RF-DETR](https://github.com/roboflow/rf-detr) — model architecture and pretrained weights
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt) — optimised GPU inference
- [OpenCV](https://github.com/opencv/opencv) — image I/O and visualisation
- [YOLOs-CPP-TensorRT](https://github.com/Geekgineer/YOLOs-CPP-TensorRT) — TensorRT C++ pipeline inspiration

---

<p align="center">
  <strong>⭐ If RF-DETR C++ helps your project, consider giving it a star!</strong>
</p>
