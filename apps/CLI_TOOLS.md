# RF-DETR C++ CLI Tools

Command-line tools for building engines, running inference, and benchmarking.
All binaries are built automatically when `RFDETR_BUILD_APPS=ON` (the default).
See the [main README](../README.md#build-from-source) for build instructions.

---

## rfdetr_build

Convert an ONNX model to a TensorRT engine.

```sh
# FP32
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp32

# FP16
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx --precision fp16

# INT8 (TRT 11+, QDQ path)
./build/rfdetr_build \
    --onnx   trt-files/onnx/rf-detr-nano-int8.onnx \
    --engine trt-files/onnx/rf-detr-nano-int8.engine \
    --precision fp32

# Dynamic batch (max 4 images) — the ONNX must be exported with --dynamic-batch,
# otherwise its batch dimension is baked to 1 and this build fails.
python trt-files/scripts/export_onnx.py --variant nano --dynamic-batch \
    --out-dir trt-files/onnx --name rf-detr-nano-dyn
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano-dyn.onnx --precision fp16 \
    --min-batch 1 --opt-batch 4 --max-batch 4
```

TensorRT tunes kernels for `--opt-batch`, so set it to the batch size you will
actually run. A dynamic engine is slightly slower at batch 1 than a static one —
keep both if you need lowest single-frame latency as well as batch throughput.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--onnx` | required | Path to input ONNX file |
| `--engine` | `<onnx>.engine` | Output engine path |
| `--meta` | `<engine>.json` | Output sidecar JSON path |
| `--precision` | `fp32` | `fp32`, `fp16`, or `int8` |
| `--calib` | | Calibration cache (TRT < 11 INT8) |
| `--workspace-mib` | `4096` | Builder workspace in MiB |
| `--min-batch` | `1` | Minimum batch size |
| `--opt-batch` | `1` | Optimal batch size |
| `--max-batch` | `1` | Maximum batch size |
| `--verbose` | off | Verbose TRT builder output |

---

## rfdetr_detect

Run detection on a single image and save an annotated result.

```sh
./build/rfdetr_detect \
    --engine trt-files/onnx/rf-detr-nano-fp16.engine \
    --image  asset/test_img.jpg \
    --out    out/result.jpg \
    --threshold 0.5
```

---

## rfdetr_seg

Run instance segmentation on a single image.

```sh
./build/rfdetr_seg \
    --engine trt-files/onnx/rf-detr-seg-nano-fp16.engine \
    --image  asset/test_img.jpg \
    --out    out/seg_result.jpg \
    --threshold 0.5
```

---

## rfdetr_video

Run detection on a video file or live camera stream.

```sh
# Video file
./build/rfdetr_video \
    --engine trt-files/onnx/rf-detr-nano-fp16.engine \
    --input  video.mp4 \
    --out    out/result.mp4

# Camera (device index 0)
./build/rfdetr_video \
    --engine trt-files/onnx/rf-detr-nano-fp16.engine \
    --input  0
```

---

## rfdetr_batch

Time batched detection throughput. Takes **one** image and replicates it `--batch N`
times — it is a throughput benchmark, not a tool for processing a set of images.
Requires a **dynamic-batch** engine when `N > 1`; see `rfdetr_build` above.

```sh
./build/rfdetr_batch \
    --engine trt-files/onnx/rf-detr-nano-dyn-fp16.engine \
    --image  asset/test_img.jpg \
    --batch  4 \
    --threshold 0.5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--engine` | required | Engine built with `--max-batch >= N` |
| `--image` | required | Image replicated across the batch |
| `--batch` | `2` | Batch size |
| `--threshold` | `0.5` | Score threshold |
| `--iters` | `20` | Timing iterations |
| `--out-dir` | | Write annotated `batch_0.jpg` … |

To batch *distinct* images, call `detect_batch()` / `segment_batch()` from C++
directly — see the [README](../README.md#batch-inference).

Batching trades latency for throughput: it amortizes launch overhead and fills
the GPU, but no image completes until the whole batch does. Use it for offline
work over a set of images, not for a live camera feed.

---

## rfdetr_bench

Latency and throughput benchmark. See [benchmarks/BENCHMARKING.md](../benchmarks/BENCHMARKING.md) for full usage, sweep scripts, and cross-device comparison.

```sh
# Quick run (10 warm-up, 100 iters)
./build/rfdetr_bench quick trt-files/onnx/rf-detr-nano-fp32.engine asset/test_img.jpg
```

---

## rfdetr_inspect

Print engine I/O bindings, shapes, and sidecar metadata.

```sh
./build/rfdetr_inspect trt-files/onnx/rf-detr-nano-fp16.engine
./build/rfdetr_inspect trt-files/onnx/rf-detr-nano-fp16.engine \
                       trt-files/onnx/rf-detr-nano-fp16.engine.json
```

---

## rfdetr_smoke

Load an engine, fill all inputs with zeros, run one forward pass, and print output stats.
Useful for verifying a newly built engine before running real data.

```sh
./build/rfdetr_smoke trt-files/onnx/rf-detr-nano-fp16.engine
./build/rfdetr_smoke trt-files/onnx/rf-detr-nano-fp16.engine 100   # 100 iters
```

---

## rfdetr_run

Low-level end-to-end test that bypasses the high-level detector API and drives
`TrtSession` + `ImagePreprocessor` + `decode_detections` directly.
Useful for debugging the inference pipeline step by step.

```sh
./build/rfdetr_run \
    --engine trt-files/onnx/rf-detr-nano-fp16.engine \
    --image  asset/test_img.jpg \
    --iters  10
```
