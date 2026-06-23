# User Journey

What it looks like to *use* `rf-detr-cpp` end-to-end. No implementation details
— just the commands, files, and outputs from a user's perspective.

> **Current status (2026-06-20):** Steps 0 / 1 / 2 / 2.5 are implemented. So
> sections 1, 2, 3, and 4 below are real today. Sections 5 onward describe the
> final shape of the library and are aspirational until the corresponding
> implementation steps land.

---

## 1. Install once

```sh
# CUDA toolkit + driver (one-time, system-wide)
# - NVIDIA driver capable of CUDA 12.x
# - CUDA Toolkit 12.x at /usr/local/cuda

# TensorRT 11.x (Ubuntu 24.04 example, see docs/install_tensorrt.md)
sudo apt-get install -y tensorrt tensorrt-dev

# OpenCV
sudo apt-get install -y libopencv-dev
```

That's all the system-wide setup. No Python required to run inference.

## 2. Get the source and build

```sh
git clone <this repo>
cd rf-detr-cpp

cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=120   # 87 Orin / 101 Thor / 120 Blackwell
cmake --build build -j
```

You'll get four binaries in `build/`:

| Binary           | What it does                                                |
|------------------|-------------------------------------------------------------|
| `rfdetr_build`   | ONNX → TensorRT engine (pure C++)                           |
| `rfdetr_inspect` | Print engine bindings + cross-check sidecar metadata        |
| `rfdetr_smoke`   | Run a single forward pass with zeros (smoke test)           |
| `rfdetr_image` * | Run detection (or seg) on an image, save annotated output   |
| `rfdetr_video` * | Run detection (or seg) on a video, save annotated output    |
| `rfdetr_camera`* | Live camera demo                                            |
| `rfdetr_batch` * | Batched inference over a directory of images                |
| `rfdetr_bench` * | Latency / throughput / accuracy benchmark harness           |

\* lands in later steps.

The project is also a library — link your own apps against `rfdetr::rfdetr`.

## 3. Get a model

Two options:

### Option A — download a pre-exported ONNX (most users)

Roboflow publishes `.onnx` exports for their detection variants. Download one
and drop it into `trt-files/onnx/`. This path **doesn't require Python**.

### Option B — export from PyTorch yourself

Only needed if you've fine-tuned RF-DETR on your own dataset, or want a
variant Roboflow hasn't pre-exported.

```sh
# One-time Python venv. Use uv or plain pip — example uses uv.
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r trt-files/scripts/requirements.txt

# Export.
# First run downloads the pretrained checkpoint to ~/.roboflow/models/<variant>.pth
# (e.g. ~349 MB for nano). Future exports of the same variant reuse the cache.
python trt-files/scripts/export_onnx.py --variant nano --out-dir trt-files/onnx
# -> trt-files/onnx/rf-detr-nano.onnx
# -> trt-files/onnx/rf-detr-nano.json   (metadata sidecar)
```

> The `.pth` checkpoint stays in `~/.roboflow/models/` — the export script
> doesn't copy it into the project tree. That cache is shared across projects
> and survives venv recreation.

The sidecar JSON next to the ONNX captures everything the runtime needs:
variant identity, input H×W, normalization stats, color order, num_queries,
num_classes, has_masks, etc. You don't edit it; the export tool writes it.

## 4. Build a TensorRT engine (pure C++)

```sh
./build/rfdetr_build --onnx trt-files/onnx/rf-detr-nano.onnx
```

Output:

```
[rfdetr_build] onnx        : trt-files/onnx/rf-detr-nano.onnx
[rfdetr_build] engine out  : trt-files/onnx/rf-detr-nano-fp32.engine
[rfdetr_build] precision   : fp32
[rfdetr_build] workspace   : 4096 MiB
[rfdetr_build] meta in     : trt-files/onnx/rf-detr-nano.json (variant=nano, 384x384)
[rfdetr_build] parsed ONNX, 1 input(s), 2 output(s)
[rfdetr_build] input 'input' shape=[1,3,384,384]
[rfdetr_build] building... (this can take 10-60s)
[rfdetr_build] built in 12.4s, plan size = 105.4 MiB
[rfdetr_build] wrote trt-files/onnx/rf-detr-nano-fp32.engine
[rfdetr_build] wrote trt-files/onnx/rf-detr-nano-fp32.engine.json
```

The output filename is automatically tagged with the precision so you can keep
multiple builds of the same model side-by-side:

```
trt-files/onnx/
├── rf-detr-nano.onnx
├── rf-detr-nano.json                       (ONNX sidecar)
├── rf-detr-nano-fp32.engine                (FP32 build)
├── rf-detr-nano-fp32.engine.json
├── rf-detr-nano-fp16.engine                (FP16 build, when ONNX is FP16)
└── rf-detr-nano-fp16.engine.json
```

Common variants:

```sh
# FP16 (requires an FP16 ONNX — see note below):
./build/rfdetr_build --onnx model.onnx --precision fp16

# Dynamic batch up to 8:
./build/rfdetr_build --onnx model.onnx --max-batch 8 --opt-batch 4

# Bigger workspace if the build complains about memory:
./build/rfdetr_build --onnx model.onnx --workspace-mib 8192

# Custom engine path (overrides the auto -<precision> suffix):
./build/rfdetr_build --onnx model.onnx --engine /tmp/my.engine

# Verbose TRT logs while debugging:
./build/rfdetr_build --onnx model.onnx --verbose
```

> **About `--precision fp16`:** TensorRT 11+ networks are strongly-typed —
> the actual numeric precision is fixed by the ONNX tensor dtypes, not by a
> builder flag. The `--precision` arg controls the engine filename and the
> sidecar label, but the engine's real precision matches the ONNX. To get a
> genuine FP16 engine, re-export the model from PyTorch with FP16 weights or
> insert Cast ops at the network boundaries. INT8 with calibration arrives
> in step 9.

> **Engines are GPU-specific.** A `.engine` built on RTX 5070 Ti won't load on
> Jetson Orin and vice versa. Build once per device.

## 5. Verify the engine

```sh
./build/rfdetr_inspect trt-files/onnx/rf-detr-nano-fp32.engine \
                       trt-files/onnx/rf-detr-nano-fp32.engine.json
```

Output:

```
engine: trt-files/onnx/rf-detr-nano-fp32.engine
  trt header: 11.1.0
  bindings:
    [0] input         INPUT   fp32   shape=[1,3,384,384]
    [1] dets          OUTPUT  fp32   shape=[1,300,4]
    [2] labels        OUTPUT  fp32   shape=[1,300,91]

meta sidecar: trt-files/onnx/rf-detr-nano-fp32.engine.json
  variant      = nano
  input_hxw    = 384x384
  num_queries  = 300
  num_classes  = 90
  has_masks    = false
  color_order  = RGB
  precision    = fp32
  mean         = [0.4850, 0.4560, 0.4060]
  std          = [0.2290, 0.2240, 0.2250]
```

If anything mismatches, you get a `WARNING:` line.

## 6. Smoke test (sanity check the engine actually executes)

```sh
./build/rfdetr_smoke trt-files/onnx/rf-detr-nano-fp32.engine 10
```

Output:

```
loaded: trt-files/onnx/rf-detr-nano-fp32.engine
  inputs:  1
  outputs: 2
  IN  input         dtype=fp32   bytes=1769472   shape=[1,3,384,384]
  OUT dets          dtype=fp32   bytes=4800      shape=[1,300,4]
  OUT labels        dtype=fp32   bytes=109200    shape=[1,300,91]
running infer x10...
infer OK  total=24.0 ms  mean=2.4 ms/iter
  out dets    fp32  n=1200    mean=+0.441   max|x|=1.073
  out labels  fp32  n=27300   mean=-6.387   max|x|=9.111
```

Feeds zeros into the network. The output values are meaningless, but the time
is real — that's your engine's per-frame cost minus image preprocessing.

---

## 7. Detect objects in an image *(Step 5 onward)*

```sh
./build/rfdetr_run \
    --engine trt-files/onnx/rf-detr-nano-fp32.engine \
    --image  /path/to/photo.jpg \
    --threshold 0.5 \
    --out     out/photo.annotated.jpg
```

Output:

```
loaded engine: ...nano.engine (variant=nano, 384x384)
photo.jpg: 5 detections in 3.1 ms
  person     0.94  bbox=[ 412,  82, 631, 480]
  dog        0.87  bbox=[ 712, 240, 956, 503]
  bicycle    0.71  bbox=[ 130, 280, 405, 510]
  ...
wrote out/photo.annotated.jpg
```

The annotated JPEG has boxes + class labels drawn on it. JSON output via
`--json out/photo.json` if you want machine-readable detections.

For segmentation:

```sh
./build/rfdetr_image \
    --engine trt-files/onnx/rf-detr-seg-medium.engine \
    --image  /path/to/photo.jpg \
    --seg \
    --out    out/photo.annotated.jpg \
    --masks  out/masks/
```

Each detection gets a colored mask overlay; binary masks per detection are
written to `out/masks/<image>.<idx>.png`.

## 8. Detect on a video stream *(Step 7)*

```sh
./build/rfdetr_video \
    --engine trt-files/onnx/rf-detr-small.engine \
    --input  input.mp4 \
    --output out.mp4 \
    --threshold 0.5
```

Console shows live FPS:

```
input.mp4: 1920x1080 @ 30.0 fps, 4500 frames
processing... [#####.....]   45% | 11.2s elapsed | 2350 frames | 209 fps
```

Pipelined mode (decode + infer + draw on separate threads) for higher FPS:

```sh
./build/rfdetr_video --pipelined ...
```

Live camera:

```sh
./build/rfdetr_camera --engine ...nano.engine --device /dev/video0
```

## 9. Use the library from your own C++ code *(Step 5+)*

```cpp
#include <rfdetr/tasks/detector.hpp>
#include <opencv2/opencv.hpp>

int main() {
    rfdetr::tasks::Detector det("rf-detr-nano.engine");

    cv::Mat img = cv::imread("photo.jpg");
    auto results = det.detect(img, /*threshold=*/0.5);

    for (const auto& d : results) {
        std::cout << d.class_id << " " << d.score
                  << " [" << d.box.x1 << "," << d.box.y1 << ","
                          << d.box.x2 << "," << d.box.y2 << "]\n";
    }
}
```

CMake:

```cmake
find_package(rfdetr CONFIG REQUIRED)
add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE rfdetr::rfdetr ${OpenCV_LIBS})
```

Segmentation is the same shape:

```cpp
#include <rfdetr/tasks/segmenter.hpp>

rfdetr::tasks::Segmenter seg("rf-detr-seg-medium.engine");
auto results = seg.segment(img, /*threshold=*/0.5);
for (const auto& r : results) {
    cv::imshow("mask", r.mask);  // CV_8UC1, original-image size
}
```

## 10. Benchmark on your hardware *(Step 13)*

```sh
./build/rfdetr_bench \
    --engine trt-files/onnx/rf-detr-small.engine \
    --device "rtx-5070ti" \
    --warmup 50 --iters 500 \
    --report results/rtx-5070ti_small_fp16_b1.json
```

Output (also written to JSON):

```
device: rtx-5070ti  variant: small  precision: fp16  batch: 1
warmup 50 iters... done
measure 500 iters... done
  latency_ms      mean=2.41   p50=2.39   p90=2.51   p99=2.78   min=2.31  max=3.12
  gpu_time_ms     mean=2.06   p99=2.31
  throughput      415.7 fps
  cuda_graph      enabled
```

Compare across devices:

```sh
./benchmarks/scripts/compare_devices.py results/*.json > results/REPORT.md
```

Produces a Markdown table you can drop into release notes.

---

## 11. Deploying

### Bare-metal Linux

You ship:

- The four C++ binaries (`build/rfdetr_*`) **or** your own app linked against
  `librfdetr.so`
- The `.engine` file you built for the target GPU
- The `.engine.json` sidecar

No Python. No PyTorch. No trtexec. Just CUDA + TensorRT runtime + OpenCV
already on the box.

### Docker *(Step 12)*

```sh
docker build -t rfdetr:latest -f docker/Dockerfile.x86_64 .
docker run --rm --gpus all \
    -v $(pwd)/trt-files:/work/engines \
    rfdetr:latest \
    rfdetr_image --engine /work/engines/rf-detr-nano.engine --image /work/sample.jpg
```

Jetson:

```sh
docker build -t rfdetr:jetson -f docker/Dockerfile.jetson .
docker run --rm --runtime nvidia rfdetr:jetson rfdetr_smoke /work/engine.plan
```

## 12. The full pipeline at a glance

```
                   ┌──────────────────────┐
                   │ upstream rfdetr (PT) │   one-shot, Python only
                   └──────────┬───────────┘
                              │ export_onnx.py        (or download a published ONNX)
                              ▼
                   ┌──────────────────────┐
                   │  rf-detr-X.onnx      │
                   │  rf-detr-X.json      │   ◄── metadata sidecar
                   └──────────┬───────────┘
                              │ rfdetr_build  (PURE C++)
                              ▼
                   ┌──────────────────────┐
                   │  rf-detr-X.engine    │
                   │  rf-detr-X.engine.json│
                   └──────────┬───────────┘
                              │
            ┌─────────────────┼────────────────┬─────────────────┐
            ▼                 ▼                ▼                 ▼
       rfdetr_image     rfdetr_video    rfdetr_camera     your own C++ app
                                                          (#include <rfdetr/...>)
```

Two Python touches, both optional and one-shot:

- Get the ONNX (or download a pre-exported one).
- Run INT8 calibration if you want INT8 (Step 9).

Everything in the steady-state inference path is C++.
