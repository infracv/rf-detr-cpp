# Benchmarking Guide

How to reproduce the published numbers and run your own benchmarks. This guide only covers FP32 and FP16 — see [INT8_QUANTIZATION.md](../trt-files/INT8_QUANTIZATION.md) if you need INT8.

---

# NVIDIA GPU (Desktop / Server)

## Prerequisites

Build with `-DRFDETR_BUILD_BENCHMARKS=ON` (see the [main README](../README.md#build-from-source) for the full build command and GPU arch table).

You also need at least one built engine. See `trt-files/scripts/` for ONNX export and `rfdetr_build` for engine conversion.

## Single Engine, Single Run

```sh
# Quick sanity check (10 warm-up, 100 iters)
./build/rfdetr_bench quick trt-files/onnx/rf-detr-nano-fp16.engine asset/test_img.jpg

# Full run with JSON output
./build/rfdetr_bench image trt-files/onnx/rf-detr-nano-fp16.engine asset/test_img.jpg \
    --device rtx-4090 \
    --warmup 50 \
    --iters  500 \
    --json \
    --output-dir benchmarks/results
```

The `--device` label is written into the JSON report and used for filenames — use a short slug like `rtx-4090` or `a100-sxm`.

For segmentation engines, task type is detected automatically from the engine sidecar (`.engine.json`):

```sh
./build/rfdetr_bench image trt-files/onnx/rf-detr-seg-nano-fp16.engine asset/test_img.jpg \
    --device rtx-4090 --warmup 50 --iters 500 --json --output-dir benchmarks/results
```

## Sweep All Engines

Use `run_all.sh` to benchmark every `.engine` file in a directory in one shot:

```sh
./benchmarks/scripts/run_all.sh \
    --engine-dir trt-files/onnx \
    --image      asset/test_img.jpg \
    --device     rtx-4090 \
    --out-dir    benchmarks/results \
    --warmup     50 \
    --iters      500
```

The script writes one JSON report per engine to `--out-dir`, then calls `compare_devices.py` automatically to print a summary table.

## Compare Results Across Devices

Collect JSON reports from multiple machines into the same directory, then run:

```sh
python3 benchmarks/scripts/compare_devices.py benchmarks/results/

# CSV output
python3 benchmarks/scripts/compare_devices.py benchmarks/results/ --csv > table.csv
```

The script groups rows by device, variant, precision, and task and renders a Markdown table with FPS, mean latency, P50, P99, and GPU memory.

## Video Throughput

```sh
./build/rfdetr_bench video trt-files/onnx/rf-detr-nano-fp16.engine asset/test_video.mp4 \
    --device rtx-4090 \
    --iters  500 \
    --json \
    --output-dir benchmarks/results
```

## Tips

- **Pin the GPU clock** before benchmarking for reproducible numbers:
  ```sh
  sudo nvidia-smi -pm 1
  sudo nvidia-smi --lock-gpu-clocks=<base>,<max>
  ```
- Run with `compute-sanitizer` after any CUDA change to catch memory errors before benchmarking:
  ```sh
  compute-sanitizer --tool memcheck ./build/rfdetr_bench quick <engine> <image>
  ```
- If `rfdetr_bench` is not in `PATH`, pass it explicitly with `--bench ./build/rfdetr_bench` to `run_all.sh`.
- Results in `benchmarks/results/` are gitignored for large JSON files — only `benchmark_results.csv` is committed.

---

# NVIDIA Jetson (Orin / Thor)

Everything in the section above applies on Jetson too. This section only covers what's different.

## Compute Capability

| Device | `CMAKE_CUDA_ARCHITECTURES` |
|:-------|:--------------------------:|
| Jetson Orin (Orin NX, Orin Nano, AGX Orin) | `87` |
| Jetson AGX Thor | `101` |

Build with the correct value explicitly, do not rely on the multi-arch default:

```sh
# Orin NX 16GB
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=87
cmake --build build -j$(nproc)

# AGX Thor
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=101
cmake --build build -j$(nproc)
```

## JetPack / TensorRT Version

Check what ships on your device before building engines — engine files are TRT-version-specific and won't transfer between JetPack releases.

```sh
dpkg -l | grep -i tensorrt
cat /etc/nv_tegra_release
```

Export ONNX and build the `.engine` file directly on the Jetson (or on a build host running the identical JetPack/TRT version). An engine built on a desktop RTX GPU will not load on Jetson.

## Lock Clocks Before Benchmarking

Jetson power modes and dynamic clock scaling introduce significant run-to-run variance if left on defaults. Lock clocks to the maximum for reproducible numbers:

```sh
# Set max performance power mode (MAXN / MAXN SUPER, name varies by JetPack release)
sudo nvpmodel -m 0

# Lock GPU, CPU, and EMC clocks to their max values
sudo jetson_clocks

# Confirm current clocks
sudo jetson_clocks --show
```

Run this before every benchmark session — a reboot or power-mode change resets it.

## Unified Memory Caveat

Jetson uses a unified CPU/GPU memory pool (no dedicated VRAM). `cudaMemGetInfo`, which `rfdetr_bench` uses for the "GPU Memory" column, reports free/total *system* memory, not a GPU-only allocation. Numbers are still useful for relative comparison across precisions, but are not directly comparable to discrete-GPU results in the main benchmark table.

For a more accurate picture of memory pressure on Jetson, cross-check with:

```sh
tegrastats
```

Run it in a second terminal while `rfdetr_bench` is active.

## Running the Benchmark

Same commands as desktop, just label the device correctly so results don't get mixed up in `compare_devices.py`:

```sh
# Orin NX 16GB
./build/rfdetr_bench image trt-files/onnx/rf-detr-nano-fp16.engine asset/test_img.jpg \
    --device jetson-orin-nx-16gb \
    --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results

# AGX Thor
./build/rfdetr_bench image trt-files/onnx/rf-detr-nano-fp16.engine asset/test_img.jpg \
    --device jetson-agx-thor \
    --warmup 50 --iters 500 --json \
    --output-dir benchmarks/results
```

For a full sweep across all engines on the device:

```sh
./benchmarks/scripts/run_all.sh \
    --engine-dir trt-files/onnx \
    --image      asset/test_img.jpg \
    --device     jetson-orin-nx-16gb \
    --out-dir    benchmarks/results \
    --warmup     50 --iters 500
```

## Thermal Throttling

Jetson modules can throttle under sustained load, especially in fanless or passively-cooled carrier boards. If you see FPS drop over a long `--iters` run:

- Check `tegrastats` for `thermal` or throttle warnings.
- Ensure the fan/heatsink is attached and `nvpmodel` allows the fan to run at full speed.
- Prefer shorter, repeated runs over one very long run if throttling is unavoidable in your enclosure, and note the ambient temperature in your results.

## Reporting Results

When sharing Jetson numbers (e.g. in an issue or PR), include:

- Exact module (Orin NX 16GB vs 8GB, Thor variant)
- JetPack version and TensorRT version
- Power mode used (`nvpmodel -m <N>`) and whether `jetson_clocks` was applied
- Carrier board / cooling setup if throttling is a concern

---

## What the Numbers Mean

| Metric | Source | Notes |
|--------|--------|-------|
| FPS | `1000 / mean_latency_ms` | Full pipeline: pre + infer + post |
| Avg Latency | `std::chrono::steady_clock` | Wall time per frame |
| P50 / P99 | sorted latency samples | Measures jitter, not just average |
| GPU Memory | `cudaMemGetInfo` delta | Peak allocation during one forward pass; on Jetson this reflects unified system memory, not dedicated VRAM |

Warm-up iterations are discarded. The default of 50 warm-up + 500 measured iterations gives stable P99 estimates for most engines.
