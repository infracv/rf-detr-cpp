# Benchmarking Guide

How to reproduce the published numbers and run your own benchmarks. This guide only covers FP32 and FP16 — see [INT8_QUANTIZATION.md](../trt-files/INT8_QUANTIZATION.md) if you need INT8.

> **Running this yourself?** Always pass `--csv-name your_benchmark.csv` to `rfdetr_bench` (or `--csv-name` to `run_all.sh`). Without it, results append to the committed `benchmark_results.csv`, mixing your numbers with the published ones.

---

# Setup (Common to All Devices)

Do this once per device, whether it's a desktop GPU or a Jetson.

## 1. Install Dependencies

```sh
# Python deps for ONNX export (uv recommended)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

C++ build dependencies (CUDA, TensorRT, OpenCV, CMake) are covered in the [main README](../README.md#prerequisites).

## 2. Build the Library with Benchmarks Enabled

```sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=<arch> \
    -DRFDETR_BUILD_BENCHMARKS=ON
cmake --build build -j$(nproc)
```

Replace `<arch>` with your device's compute capability (see the [GPU arch table](../README.md#build-from-source) — desktop RTX cards, or `87`/`101` for Jetson, covered below).

## 3. Export ONNX (Detection + Segmentation)

```sh
python trt-files/scripts/export_onnx.py --variant nano     --out-dir trt-files/onnx
python trt-files/scripts/export_onnx.py --variant seg-nano --out-dir trt-files/onnx
```

This produces the ONNX files for both tasks. Do this once per model variant, it does not need to be repeated per device.

## 4. Build All 4 Engines (FP32 + FP16, Detection + Segmentation)

Engines are GPU/driver/TRT-version specific, so this step must be run **on the target device**, not copied over from another machine.

```sh
# Detection — FP32
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp32

# Detection — FP16
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp16

# Segmentation — FP32
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-seg-nano.onnx --precision fp32

# Segmentation — FP16
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-seg-nano.onnx --precision fp16
```

This gives you:

```
trt-files/onnx/rf-detr-nano-fp32.engine
trt-files/onnx/rf-detr-nano-fp16.engine
trt-files/onnx/rf-detr-seg-nano-fp32.engine
trt-files/onnx/rf-detr-seg-nano-fp16.engine
```

Verify they're all there before benchmarking:

```sh
ls trt-files/onnx/*.engine
```

---

# NVIDIA GPU (Desktop / Server)

## Pin the GPU Clock

For reproducible numbers, lock the clock before benchmarking:

```sh
sudo nvidia-smi -pm 1
sudo nvidia-smi --lock-gpu-clocks=<base>,<max>
```

## Run All 4 Benchmarks

```sh
./build/rfdetr_bench image trt-files/onnx/rf-detr-nano-fp32.engine asset/test_img.jpg \
    --device rtx-4090 --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results --csv-name your_benchmark.csv

./build/rfdetr_bench image trt-files/onnx/rf-detr-nano-fp16.engine asset/test_img.jpg \
    --device rtx-4090 --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results --csv-name your_benchmark.csv

./build/rfdetr_bench image trt-files/onnx/rf-detr-seg-nano-fp32.engine asset/test_img.jpg \
    --device rtx-4090 --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results --csv-name your_benchmark.csv

./build/rfdetr_bench image trt-files/onnx/rf-detr-seg-nano-fp16.engine asset/test_img.jpg \
    --device rtx-4090 --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results --csv-name your_benchmark.csv
```

Task type (detection vs segmentation) is auto-detected from the engine sidecar (`.engine.json`). Use a short `--device` slug like `rtx-4090` or `a100-sxm` — it's written into filenames and JSON reports. `--csv-name` keeps your results in their own file instead of appending to the committed `benchmark_results.csv`.

## Sweep All Engines in One Shot

```sh
./benchmarks/scripts/run_all.sh \
    --engine-dir trt-files/onnx \
    --image      asset/test_img.jpg \
    --device     rtx-4090 \
    --out-dir    benchmarks/results \
    --warmup     50 \
    --iters      500 \
    --csv-name   your_benchmark.csv
```

This benchmarks every `.engine` file under `--engine-dir` and calls `compare_devices.py` automatically at the end.

## Video Throughput

```sh
./build/rfdetr_bench video trt-files/onnx/rf-detr-nano-fp16.engine asset/test_video.mp4 \
    --device rtx-4090 \
    --iters  500 \
    --json \
    --output-dir benchmarks/results
```

## Tips

- Run with `compute-sanitizer` after any CUDA change to catch memory errors before benchmarking:
  ```sh
  compute-sanitizer --tool memcheck ./build/rfdetr_bench quick <engine> <image>
  ```
- If `rfdetr_bench` is not in `PATH`, pass it explicitly with `--bench ./build/rfdetr_bench` to `run_all.sh`.
- Results in `benchmarks/results/` are gitignored for large JSON files — only `benchmark_results.csv` is committed.

---

# NVIDIA Jetson (Orin / Thor)

Everything in the setup section above applies here too — same ONNX export, same 4 engines, built on-device. This section covers what's different on Jetson.

## Compute Capability

| Device | `CMAKE_CUDA_ARCHITECTURES` |
|:-------|:--------------------------:|
| Jetson Orin (Orin NX, Orin Nano, AGX Orin) | `87` |
| Jetson AGX Thor | `101` |

```sh
# Orin NX 16GB
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc)

# AGX Thor
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=101
cmake --build build -j$(nproc)
```

## Check JetPack / TensorRT Version

```sh
dpkg -l | grep -i tensorrt
cat /etc/nv_tegra_release
```

Engines are TRT-version-specific and won't transfer across JetPack releases, or from a desktop RTX GPU. Build them on the Jetson itself (step 4 in Setup, above).

## Set Max Performance Mode and Lock Clocks

Jetson power modes and dynamic clock scaling introduce significant run-to-run variance if left on defaults. Do this once per boot, before benchmarking:

```sh
# Set max performance power mode (MAXN / MAXN SUPER, name varies by JetPack release)
sudo nvpmodel -m 0

# Lock GPU, CPU, and EMC clocks to their max values
sudo jetson_clocks

# Confirm current clocks
sudo jetson_clocks --show
```

No reboot is required — both commands take effect immediately. You do need to re-run them after a reboot or a power-mode change, since those reset the clocks back to default.

## Run All 4 Benchmarks

```sh
./build/rfdetr_bench image trt-files/onnx/rf-detr-nano-fp32.engine asset/test_img.jpg \
    --device jetson-orin-nx-16gb --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results --csv-name your_benchmark.csv

./build/rfdetr_bench image trt-files/onnx/rf-detr-nano-fp16.engine asset/test_img.jpg \
    --device jetson-orin-nx-16gb --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results --csv-name your_benchmark.csv

./build/rfdetr_bench image trt-files/onnx/rf-detr-seg-nano-fp32.engine asset/test_img.jpg \
    --device jetson-orin-nx-16gb --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results --csv-name your_benchmark.csv

./build/rfdetr_bench image trt-files/onnx/rf-detr-seg-nano-fp16.engine asset/test_img.jpg \
    --device jetson-orin-nx-16gb --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results --csv-name your_benchmark.csv
```

Use `--device jetson-agx-thor` (or your specific module name) on Thor. `--csv-name` keeps your results in their own file — without it, all 4 runs would append to the committed `benchmark_results.csv`.

Or sweep all engines at once:

```sh
./benchmarks/scripts/run_all.sh \
    --engine-dir trt-files/onnx \
    --image      asset/test_img.jpg \
    --device     jetson-orin-nx-16gb \
    --out-dir    benchmarks/results \
    --warmup     50 --iters 500 \
    --csv-name   your_benchmark.csv
```

## Unified Memory Caveat

Jetson uses a unified CPU/GPU memory pool (no dedicated VRAM). `cudaMemGetInfo`, which `rfdetr_bench` uses for the "GPU Memory" column, reports free/total *system* memory, not a GPU-only allocation. Numbers are still useful for relative comparison across precisions, but are not directly comparable to discrete-GPU results in the main benchmark table.

For a more accurate picture of memory pressure, cross-check with `tegrastats` in a second terminal while `rfdetr_bench` is running.

## Thermal Throttling

Jetson modules can throttle under sustained load, especially in fanless or passively-cooled carrier boards. If FPS drops over a long `--iters` run:

- Check `tegrastats` for thermal/throttle warnings.
- Ensure the fan/heatsink is attached and `nvpmodel` allows the fan to run at full speed.
- Prefer shorter, repeated runs over one very long run if throttling is unavoidable, and note the ambient temperature in your results.

## Reporting Results

When sharing Jetson numbers, include:

- Exact module (Orin NX 16GB vs 8GB, Thor variant)
- JetPack version and TensorRT version
- Power mode used (`nvpmodel -m <N>`) and whether `jetson_clocks` was applied
- Carrier board / cooling setup if throttling is a concern

---

## Compare Results Across Devices

Collect JSON reports from multiple machines into the same directory, then run:

```sh
python3 benchmarks/scripts/compare_devices.py benchmarks/results/

# CSV output
python3 benchmarks/scripts/compare_devices.py benchmarks/results/ --csv > table.csv
```

The script groups rows by device, variant, precision, and task and renders a Markdown table with FPS, mean latency, P50, P99, and GPU memory.

---

## What the Numbers Mean

| Metric | Source | Notes |
|--------|--------|-------|
| FPS | `1000 / mean_latency_ms` | Full pipeline: pre + infer + post |
| Avg Latency | `std::chrono::steady_clock` | Wall time per frame |
| P50 / P99 | sorted latency samples | Measures jitter, not just average |
| GPU Memory | `cudaMemGetInfo` delta | Peak allocation during one forward pass; on Jetson this reflects unified system memory, not dedicated VRAM |

Warm-up iterations are discarded. The default of 50 warm-up + 500 measured iterations gives stable P99 estimates for most engines.
