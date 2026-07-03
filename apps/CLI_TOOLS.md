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

# Dynamic batch (max 4 images)
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx \
    --min-batch 1 --opt-batch 2 --max-batch 4
```

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

Run detection on multiple images in a single engine call.
Requires an engine built with `--max-batch N`.

```sh
./build/rfdetr_batch \
    --engine trt-files/onnx/rf-detr-nano-fp16.engine \
    --images img1.jpg img2.jpg img3.jpg \
    --threshold 0.5
```

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
