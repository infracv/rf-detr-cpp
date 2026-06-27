# Benchmarking Guide

How to reproduce the published numbers and run your own benchmarks on any GPU.

---

## Prerequisites

Build with `-DRFDETR_BUILD_BENCHMARKS=ON` (see the [main README](../README.md#build-from-source) for the full build command and GPU arch table).

You also need at least one built engine. See `trt-files/scripts/` for ONNX export and `rfdetr_build` for engine conversion.

---

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

---

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

## Video Throughput

```sh
./build/rfdetr_bench video trt-files/onnx/rf-detr-nano-fp16.engine asset/test_video.mp4 \
    --device rtx-4090 \
    --iters  500 \
    --json \
    --output-dir benchmarks/results
```

---

## What the Numbers Mean

| Metric | Source | Notes |
|--------|--------|-------|
| FPS | `1000 / mean_latency_ms` | Full pipeline: pre + infer + post |
| Avg Latency | `std::chrono::steady_clock` | Wall time per frame |
| P50 / P99 | sorted latency samples | Measures jitter, not just average |
| GPU Memory | `cudaMemGetInfo` delta | Peak allocation during one forward pass |

Warm-up iterations are discarded. The default of 50 warm-up + 500 measured iterations gives stable P99 estimates for most engines.

---

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
