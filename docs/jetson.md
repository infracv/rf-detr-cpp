# Jetson deployment guide

Covers Orin NX 16 GB (CC 8.7) and AGX Thor (CC 10.1) running JetPack 6.1+.

---

## JetPack / TRT version matrix

| JetPack | L4T    | CUDA | TRT   | Notes |
|---------|--------|------|-------|-------|
| 6.1     | 36.3   | 12.4 | 10.3  | Minimum supported |
| 6.2     | 36.4   | 12.6 | 10.6  | Recommended for Orin |
| 7.0+    | 37.x   | 12.8 | 11.x  | Required for AGX Thor |

rf-detr-cpp requires TensorRT >= 10.0. JetPack 6.0 (TRT 8.6) is not supported.

---

## Pre-build checklist

```sh
# Verify installed versions.
dpkg -l | grep nvidia-jetpack          # JetPack version
nvcc --version                         # CUDA version
/usr/src/tensorrt/bin/trtexec --version # TRT version

# Pin clocks for reproducible benchmark results.
sudo nvpmodel -m 0          # MAXN power mode
sudo jetson_clocks          # lock GPU/CPU/EMC clocks
```

---

## Native build (on-device, recommended)

```sh
# 1. Install build deps.
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake ninja-build git \
    libopencv-dev libopencv-contrib-dev \
    python3-pip python3-dev

# 2. Clone and configure.
git clone https://github.com/your-org/rf-detr-cpp.git
cd rf-detr-cpp

# Orin NX only:
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=87

# AGX Thor only:
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=101

# Both Orin + Thor in one binary:
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES="87;101"

# 3. Build.
cmake --build build -j$(nproc)
```

TensorRT is in `/usr/lib/aarch64-linux-gnu/`; CMake's `FindTensorRT.cmake`
finds it automatically. If your JetPack puts headers elsewhere:

```sh
cmake -B build -S . \
    -DTENSORRT_DIR=/usr/local/cuda \
    -DCMAKE_CUDA_ARCHITECTURES=87
```

---

## Export ONNX on-device

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python trt-files/scripts/export_onnx.py --variant nano --out-dir trt-files/onnx
```

Alternatively, export on a workstation and copy the `.onnx` + `.json` sidecar to the Jetson.
Engine files (`.engine`) are device-specific — always build the engine on the target device.

---

## Build engine on-device

```sh
# FP32 (safe baseline)
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp32

# INT8 (after generating a calibration cache — see trt-files/scripts/int8_calibrate.py)
python trt-files/scripts/int8_calibrate.py \
    --onnx   trt-files/onnx/rf-detr-nano.onnx \
    --images /path/to/calib_images \
    --output trt-files/onnx/rf-detr-nano.calib

python trt-files/scripts/build_engine.py \
    --onnx trt-files/onnx/rf-detr-nano.onnx \
    --precision int8 \
    --int8-calib trt-files/onnx/rf-detr-nano.calib
```

---

## Docker on-device

```sh
docker build -f docker/Dockerfile.jetson -t rfdetr-cpp:jetson .
docker run --runtime nvidia --rm -it \
    -v $(pwd)/trt-files:/workspace/trt-files \
    -v $(pwd)/out:/workspace/out \
    rfdetr-cpp:jetson bash
```

---

## Memory-constrained variants (Orin NX 16 GB)

Orin NX shares memory between CPU and GPU. Keep an eye on usage:

```sh
tegrastats --interval 500
```

Recommended variant/batch limits for Orin NX:

| Variant    | Resolution | Batch | Notes |
|------------|-----------|-------|-------|
| nano       | 384       | 1–4   | Fine  |
| small      | 512       | 1–2   | Fine  |
| medium     | 576       | 1     | Fine  |
| base       | 560       | 1     | Fine  |
| large      | 704       | 1     | May OOM at batch > 1 with FP32 |
| seg-nano   | 312       | 1–2   | Fine  |
| seg-medium | 432       | 1     | Fine  |
| seg-large  | 504       | 1     | Monitor memory |

---

## AGX Thor specifics

AGX Thor runs JetPack 7.x (L4T 37.x, TRT 11.x). TRT 11 uses strongly-typed
networks — precision is fixed by ONNX tensor dtypes, not builder flags.
The `rfdetr_build` binary handles this transparently.

```sh
# Confirm compute capability (should be 101).
nvidia-smi --query-gpu=compute_cap --format=csv,noheader

cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=101
cmake --build build -j$(nproc)
```
