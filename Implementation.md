# rf-detr-cpp — Implementation Plan

A C++/TensorRT inference library for [RF-DETR](https://github.com/roboflow/rf-detr), structured in spirit after [YOLOs-CPP-TensorRT](https://github.com/Geekgineer/YOLOs-CPP-TensorRT). Targets detection first, segmentation second. Targets RTX desktop GPUs and Jetson (Orin NX, AGX Thor).

---

## 1. RF-DETR — what we have to implement against

This is what the inference path actually has to do. All of it is derived from `roboflow/rf-detr` source (`src/rfdetr/detr.py`, `src/rfdetr/config.py`, `src/rfdetr/models/postprocess.py`, `src/rfdetr/models/heads/segmentation.py`, `src/rfdetr/export/_onnx/exporter.py`).

### 1.1 Variants and shapes

Detection (Apache-2.0):

| Variant | Resolution | num_queries | dec_layers | patch | Weights |
|---|---|---|---|---|---|
| Nano   | 384 | 300 | 2 | 16 | `rf-detr-nano.pth` |
| Small  | 512 | 300 | 3 | 16 | `rf-detr-small.pth` |
| Medium | 576 | 300 | 4 | 16 | `rf-detr-medium.pth` |
| Base   | 560 | 300 | 3 | 14 | `rf-detr-base.pth` |
| Large  | 704 | 300 | 4 | 16 | `rf-detr-large-2026.pth` |

Segmentation (Apache-2.0, all `segmentation_head=True`, `patch_size=12`):

| Variant | Resolution | num_queries | dec_layers |
|---|---|---|---|
| Seg-Nano    | 312 | 100 | 4 |
| Seg-Small   | 384 | 100 | 4 |
| Seg-Medium  | 432 | 200 | 5 |
| Seg-Large   | 504 | 200 | 5 |
| Seg-XL      | 624 | 300 | 6 |
| Seg-2XL     | 768 | 300 | 6 |
| Seg-Preview | 432 | 200 | 4 |

`num_classes=90` across all variants (COCO 91-id sparse, ID 0 = background, last index = "toothbrush"). Resolution must be a multiple of `patch_size`.

### 1.2 Pre-processing (fixed by upstream)

Order: PIL/np → CHW float32 in `[0,1]` (RGB) → square resize to `(R, R)` → ImageNet normalize.

```
mean = [0.485, 0.456, 0.406]   stds = [0.229, 0.224, 0.225]
```

**No letterbox.** RF-DETR uses square forced resize (`F.resize(img, (R,R))`). Boxes are scaled back to the original image dims using `target_sizes = [H_orig, W_orig]`.

### 1.3 ONNX I/O contract

Single export path: `model.export(format="onnx", ...)`.

- Input: `input` `(B, 3, R, R)` float32, ImageNet-normalized RGB CHW.
- Outputs:
  - `dets`   `(B, num_queries, 4)` — **cxcywh, normalized to [0,1]**
  - `labels` `(B, num_queries, num_classes)` — **raw logits** (focal head, sigmoid activation)
  - `masks`  `(B, num_queries, R/4, R/4)` — *segmentation only*, per-query mask logits
- `dynamic_axes = None` unless `--dynamic-batch`; default opset 17.
- **Nothing pre/postprocessing-related is in the graph.**

### 1.4 Post-processing (Python reference, `src/rfdetr/models/postprocess.py`)

1. `prob = sigmoid(logits)` (focal-style, multi-label, **not softmax**).
2. Flatten `(queries × classes)`, `topk(num_select)` — typically `num_select = num_queries`.
3. `query_idx = topk_idx // num_classes`, `label = topk_idx % num_classes`.
4. `cxcywh → xyxy`, multiply by `[W, H, W, H]`.
5. Score threshold filter (default 0.5 in `predict()`). **NMS-free.**
6. Seg only: gather `masks[query_idx]`, bilinear upsample to `(H_orig, W_orig)`, threshold at 0.

### 1.5 Segmentation head

Mask2Former-style (NOT YOLACT prototype × coefficient). Mask resolution at network output is `R/4` (e.g. Seg-Large 504 → 126), upsampled to original image size and binarized in postprocess.

### 1.6 Upstream TensorRT export

`src/rfdetr/export/_tensorrt.py` is a thin shell over `trtexec`. **FP16 only**, no INT8 path, no calibration. So if we want INT8 it has to be ours.

---

## 2. YOLOs-CPP-TensorRT — structural reference

What it actually is, so we know what to copy and what to drop:

- **Header-only C++ task layer** (`include/yolos/`) plus compiled `.cu` kernels and CLI binaries (`src/`). No `.so` to link against the public API; `#include "yolos/tasks/detection.hpp"` is enough.
- **Engines must be pre-built.** `.trt` files in, never `.onnx` in. INT8 lives in `trt-files/scripts/` (Python: calibration data generator + `convert_to_tensorrt.py --int8`).
- **One class per task** under `yolos::det`, `yolos::seg`, `yolos::pose`, `yolos::obb`, `yolos::cls`. Common base: `core::TrtSessionBase`.
- **Auto-detection by output shape** picks the YOLO generation; `YOLOVersion::Auto` is default.
- **Single fused CUDA kernel** (`core/cuda_preprocessing.cu` `letterboxNormalizeKernel`) does letterbox + BGR→RGB + `/255` + HWC→NCHW into the TRT input buffer.
- **CUDA Graph capture** in `TrtSessionBase`, gated on fixed-shape engines; dynamic-shape falls back to plain `enqueueV3`.
- **15 example binaries** (image/video/camera × 5 tasks), a separate `benchmarks/yolo_unified_benchmark` sub-project, and shell-driven Python-vs-C++ parity tests.
- **Two Dockerfiles**: `Dockerfile.tensorrt` (data-center, `--gpus all`) and `Dockerfile.tensorrt.jetson` (JetPack 6.x, `--runtime nvidia`).

What transfers cleanly to RF-DETR: `TrtSessionBase` separation, fused preprocess kernel, single CUDA stream, pinned-staging async H2D, CUDA Graph capture for fixed-shape engines (RF-DETR is fixed by default), header-only + CLI binaries layout, two Dockerfiles, separate `trt-files/scripts/` for export and INT8.

What we drop: per-version output-shape autodetect (RF-DETR variants are unambiguous from shapes/metadata), letterbox math (RF-DETR uses square resize), `/255`-only normalize (we need ImageNet mean/std), NMS, anchors/strides.

---

## 3. Existing RF-DETR C++ repos — what each is, what we'll do differently

Summary of the eight repos in `rf-detr-cpp-existing-repo.txt` plus the relevant repos found via the GitHub search.

### 3.1 `hamdiboukamcha/RF-DETR-CPP-TENSORRT`
- **Is**: Single-binary CLI demo, TRT 8.6, detection only, ~78★, abandoned (Apr 2025).
- **Reusable**: `src/preprocess.cu` `warpaffine_kernel` (real letterbox, BGR→RGB, `/255`, HWC→CHW). Apache-2.0.
- **Critical bugs**: postprocess assumes a non-canonical `output_boxes`/`output_scores` export, ignores the letterbox padding when rescaling, hardcodes class index 0.
- **We differ by**: being a library (not a single binary), using canonical `dets`/`labels` outputs, square resize *plus* letterbox option, full ImageNet normalization, correct sigmoid + top-K + cxcywh decode, INT8, segmentation, Jetson, batching.

### 3.2 `olibartfast/rf-detr-cpp-inference`
- **Is**: The most production-shaped of the existing crowd — static lib + CLI + tests + ZeroCopy ring-buffer video pipeline, dual ORT/TRT backend gated at compile time, CMake FetchContent for deps, **detection + segmentation + keypoints**, actively maintained (commits as recent as June 2026).
- **Reusable**: Decoder logic (`rfdetr_inference.cpp` — sigmoid, threshold, +1 class offset, mask resize+binarize), `tensorrt_backend.cpp` ONNX→engine cache, `export_trt.sh` Docker/trtexec FP16 builder, FetchContent CMake pattern.
- **Limits**: Square resize only, **synchronous `cudaMemcpy`** (no async stream overlap), **no INT8**, no optimization profile / dynamic shapes, no GPU preprocessing, single backend per binary, no Jetson path, modest test coverage.
- **We differ by**: native TRT C++ API only (no ORT layer), GPU fused preprocess, async H2D/compute/D2H overlap on a `cudaStream_t`, INT8 with calibration, optimization profiles for batch and resolution, Jetson examples, CUDA Graph capture.

### 3.3 `mudler/rf-detr.cpp` (a.k.a. `localai-org/rf-detr.cpp`)
- **Is**: ggml-based CPU-first runtime, not a TensorRT project. Reimplements the entire RF-DETR transformer in C++ (DINOv2 backbone, decoder, segmentation head). Distributes GGUF F32/F16/Q8_0/Q4_K weights on HuggingFace. CLI `rfdetr-cli detect|quantize`, plus a clean C ABI in `include/rfdetr.h`. Brand-new, 14★, 2 commits.
- **Reusable**: `include/rfdetr.h` is the **best public C ABI to mirror** for our library shape (opaque context, `init/detect/free`, mask buffers, log callback, `image_size`/`num_queries`/`num_classes` introspection). `src/postprocess.cpp` is a useful cross-check decoder; `src/visualize.cpp` for box+mask overlay.
- **Limits (vs us)**: wrong runtime entirely — no TRT, no `.engine`, no CUDA preprocessing.
- **We differ by**: TensorRT/CUDA path, prebuilt and runtime-built `.engine`, INT8 calibration, batching, async streams, Jetson.

### 3.4 `yeokiwi/Rf_detr_cpp`
- **Is**: Windows-first CLI demo, dual-backend factory (ORT + TRT) but **AI-generated and abandoned** (default branch literally `claude/...`, two commits, 0★). Detection only.
- **Reusable**: The abstract-base + factory split (`RFDETRDetector` + `createDetector()`) is a sensible skeleton. The `--input-size` divisible-by-patch validation.
- **Critical bug**: uses **softmax+argmax**, which is wrong for the current sigmoid-trained checkpoints.
- **We differ by**: correct sigmoid postprocess, GPU preprocess, CUDA streams, INT8, segmentation, library-not-binary, Linux-first, real maintenance.

### 3.5 `mohamedsamirx/RF-DETR-TensorRT-CPP`
- **Is**: Single-binary CLI demo, native TRT, fused CUDA preprocess kernel, detection only. ~10★, last commit Apr 2025, abandoned.
- **Reusable**: `src/preprocess.cu` `warpaffine_kernel` (the same family as 3.1), `RF_DETR::build` runtime ONNX→engine path, `logging.h`/`macros.h`/`common.h` from yolov11-tensorrt template.
- **Critical bugs**: preprocess **omits ImageNet mean/std** (only `/255`), so it's silently wrong for stock Roboflow exports. No top-K → flattens 300×91=27,300 candidates per frame; any high-score query emits one box per high-scoring class. `num_classes=80` collides with 91-class output shape.
- **We differ by**: correct preprocess (mean/std as kernel params), top-K cap, segmentation, INT8, batch, library-not-binary, no hardcoded paths.

### 3.6 `Ar-Ray-code/rf_detr_ros_cpp`
- **Is**: ROS 2 (Humble/Jazzy) wrapper, **OpenVINO Intel-only**, detection only, by a credible maintainer (also maintains YOLOX-ROS). Has CI, 3★ but real iteration — most trustworthy postprocess in the field.
- **Reusable**: filename → input-size + new-gen/legacy-mode mapping; **sigmoid + argmax + foreground filter + NMS** postprocess that explicitly cites Roboflow's `benchmark.py`; `coco_names.txt` 91→80 mapping; `dispatch header + ENABLE_<BACKEND>` macro pattern.
- **Limits**: no TRT/CUDA/Jetson, no segmentation, no INT8, CPU preprocess only.
- **We differ by**: TRT/CUDA backend (Jetson + RTX), GPU preprocess, segmentation, INT8, batching. We may *add* a ROS 2 node later, mirroring this repo's pattern.

### 3.7 `zz990099/rf_detr_cpp`
- **Is**: A detection plug-in for the author's `EasyDeployTool` framework — a `.so` consumed by an unrelated tester app. Backend-agnostic via `inference_core::BaseInferCore`. Single commit, 2★, experimental.
- **Reusable**: Decoder loop in `detection_2d/detection_2d_rf_detr/src/rf_detr.cpp` — flatten-sigmoid-sort-topK with `class_id = idx % (cls+1)`, `box_id = idx / (cls+1)` is a clean reference. Constructor exposes `candidates_num=300`, `select_topk=100`. Apache-2.0.
- **Limits**: pulls in EasyDeployTool dependency chain; no segmentation; single-image API; no engine controls visible at this layer.
- **We differ by**: standalone library with no framework dependency, native TRT C++ API, GPU preprocess, async streams, segmentation, INT8.

### 3.8 `K4HVH/RF-DETR-ONNXRuntime-CPP`
- **Is**: Modern C++20 reusable library + demo (`apps/demo.cpp`) using **ONNX Runtime with TRT/CUDA/CPU EP fallback**. Two engines: `Engine` (sync) and `PipelinedEngine` (async queue). PIMPL public header. CI on Linux + Windows. GPL-3.0.
- **Reusable**: `src/preprocess.cpp` SIMD CPU normalize/HWC→CHW (AVX-512/AVX2 fused-FMA `pixel * inv_std - mean`); OpenCV-CUDA pre-wrapped CHW-planes split (no extra copy); `src/engine.cpp` IOBinding caching, pinned host memory, argmax-before-sigmoid trick (sigmoid is monotonic so compare raw logits, sigmoid only the winner); `EngineConfig` shape; `PipelinedEngine` async queue.
- **Limits**: ONNX Runtime middleman = no direct TRT plan control. **INT8 was explicitly removed** in a recent commit. Plain `cv::resize` (stretch, not letterbox or even square-aware). No segmentation. No Jetson. **GPL-3.0** — license-incompatible for any verbatim reuse if we license differently.
- **We differ by**: native TRT C++ API (full plan control: optimization profiles, INT8 calibrator, plugin layer if needed), Jetson, segmentation, our own license.

### 3.9 Additional repos found via GitHub search

- **`ridgerun-ai/deepstream-rfdetr`** (52★, MIT, active) — DeepStream bbox parser plugin for RF-DETR. Not a library, but `deepstream_rfdetr_bbox.cpp` is a tight reference for the canonical RF-DETR output decode. We differ by being a portable library, not a DeepStream plugin.
- **`marcoslucianops/DeepStream-Yolo-Seg`** (98★) — has `config_infer_primary_rfdetr_seg.txt` and matching seg parser; **the closest existing public reference for RF-DETR-Seg output decoding in C++**. Worth reading when we wire seg.
- **`olibartfast/neuriplo-infer`** (100★, MIT, active) — multi-task multi-backend C++ app; supports `rfdetr` + `rfdetrseg`. Actual RF-DETR logic lives in sister repo `neuriplo-tasks` via FetchContent. We differ by being focused on RF-DETR + TRT, not a generic framework.
- **`olibartfast/tritonic`** (30★) — Triton client app (server-side TRT). Out of scope for an in-process library.
- **`chendao12138/RF-DETR-TensorRT-EP`** (6★) — Windows-only ORT+TRT-EP demo, subset of K4HVH.
- **`YLXA321/RF-DETR-tensorRT-deploy`** (4★) — Chinese-language end-to-end pipeline; sparse C++ TRT pre/post in `src/`.
- **`clogwog/deepstream_rt_detr`** (3★) — Jetson NX2 DeepStream demo, superseded by RidgeRun's.

### 3.10 Gap summary — what nobody in the field has done

A single library that has all of:
- Native TensorRT C++ API (not ORT-EP middleman)
- Detection **and** Mask2Former-style segmentation
- Square resize **and** optional letterbox, with correct ImageNet normalization fused in CUDA
- Top-K sigmoid postprocess with proper cxcywh→xyxy + class +1 offset
- INT8 with calibration cache
- FP16 default, dynamic batch via `IOptimizationProfile`
- CUDA Graph capture for fixed-shape engines
- Async stream overlap (H2D / compute / D2H)
- Jetson examples (Orin NX, AGX Thor)
- Library + CLI + Docker, not just a single binary

This is the gap rf-detr-cpp will fill.

---

## 4. Proposed directory structure

Header-only public task layer over a compiled CUDA core, mirroring YOLOs-CPP-TensorRT's shape.

```
rf-detr-cpp/
├── CMakeLists.txt
├── README.md
├── LICENSE                                    # Apache-2.0
├── Implementation.md                          # this file
├── .clang-format / .clang-tidy
├── .github/workflows/                         # CI: build matrix x86_64 + arm64-jetson stub
│
├── include/rfdetr/
│   ├── rfdetr.hpp                             # umbrella header
│   ├── version.hpp
│   ├── core/
│   │   ├── trt_session.hpp                    # TrtSession base: load engine, IO bindings,
│   │   │                                       # pinned host buffers, stream, CUDA Graph capture
│   │   ├── trt_logger.hpp                     # nvinfer1::ILogger wrapper
│   │   ├── cuda_check.hpp                     # CUDA/TRT error macros
│   │   ├── cuda_preprocess.cuh                # public kernel decls: square resize +
│   │   │                                       # ImageNet normalize + BGR/RGB swap + HWC->NCHW
│   │   ├── cuda_postprocess.cuh               # sigmoid + topk + cxcywh->xyxy kernels (optional)
│   │   ├── variant.hpp                        # RFDETRVariant enum + variant→meta lookup
│   │   ├── engine_meta.hpp                    # parsed metadata: input HxW, num_queries,
│   │   │                                       # num_classes, has_masks, mask_HxW
│   │   ├── types.hpp                          # Box, Detection, Mask, Result structs
│   │   └── drawing.hpp                        # box + mask overlay on cv::Mat
│   ├── tasks/
│   │   ├── detector.hpp                       # RFDetrDetector — public API for det
│   │   └── segmenter.hpp                      # RFDetrSegmenter — public API for seg
│   └── c_api.h                                # opaque-handle C ABI (for FFI bindings)
│
├── src/
│   ├── core/
│   │   ├── trt_session.cpp
│   │   ├── cuda_preprocess.cu                 # fused kernel: input cv::Mat / GPU buffer →
│   │   │                                       # (B,3,R,R) f32 ImageNet-normalized RGB CHW
│   │   ├── cuda_postprocess.cu                # optional GPU sigmoid+topk; CPU fallback exists
│   │   ├── engine_meta.cpp                    # parses metadata sidecar / introspects engine
│   │   ├── variant.cpp                        # variant table, autodetect-by-shape fallback
│   │   └── drawing.cpp
│   ├── tasks/
│   │   ├── detector.cpp                       # implements RFDetrDetector (PIMPL)
│   │   └── segmenter.cpp                      # implements RFDetrSegmenter (PIMPL)
│   └── c_api.cpp                              # C ABI implementation
│
├── apps/
│   ├── rfdetr_image.cpp                       # CLI: image → detections / mask overlay
│   ├── rfdetr_video.cpp                       # CLI: video → annotated mp4
│   ├── rfdetr_camera.cpp                      # CLI: /dev/video0 live demo
│   ├── rfdetr_batch.cpp                       # CLI: batched inference over directory
│   └── rfdetr_bench.cpp                       # CLI: benchmark harness (see §7)
│
├── trt-files/
│   ├── scripts/
│   │   ├── export_onnx.py                     # wraps roboflow rfdetr export with sane defaults
│   │   ├── build_engine.py                    # trtexec wrapper (FP32/FP16, profiles, sidecar)
│   │   ├── int8_calibrate.py                  # Python TRT INT8 calibrator using a calib dataset
│   │   └── verify_engine.py                   # numerical parity vs Python reference
│   └── README.md
│
├── tests/
│   ├── unit/                                  # GoogleTest: shape/decode unit tests
│   ├── parity/                                # C++ vs Python: per-image det IoU + class match,
│   │                                            # per-image mask IoU above threshold
│   ├── data/                                  # tiny image set + expected outputs (LFS / scripts)
│   └── CMakeLists.txt
│
├── benchmarks/
│   ├── bench_runner.cpp                       # used by rfdetr_bench
│   ├── coco_subset/                           # 100-image fixed COCO val subset
│   │   ├── images/                            # downloaded by script
│   │   └── annotations.json
│   ├── scripts/
│   │   ├── run_all.sh                         # runs full matrix, emits results.json
│   │   ├── compare_devices.py                 # loads results.json{rtx5070ti, agx-thor, orin-nx}
│   │   └── coco_eval.py                       # pycocotools mAP for accuracy validation
│   └── results/                               # device-tagged JSON outputs (gitignored)
│
├── docker/
│   ├── Dockerfile.x86_64                      # CUDA + TRT 10 base, Ubuntu 22.04
│   ├── Dockerfile.jetson                      # JetPack 6.x base for AGX Thor / Orin NX
│   ├── docker-compose.yml
│   └── README.md
│
├── examples/                                  # cmake samples consuming the installed package
│   └── minimal_detector/
│
├── docs/
│   ├── installation.md
│   ├── usage.md
│   ├── quantization.md                        # FP16/INT8/TF32 guide + calibration recipe
│   ├── jetson.md                              # Orin NX / AGX Thor specifics
│   └── design.md                              # short architectural notes
│
└── third_party/                               # FetchContent only; nothing vendored
```

Key choices:

- **Header-only task layer (`include/rfdetr/tasks/`) backed by PIMPL** so users pin their TRT/CUDA versions, but symbols stay opaque. Same shape as YOLOs-CPP-TensorRT but with PIMPL behind it (lifted from K4HVH).
- **Engine sidecar metadata** (`<engine>.json`): `{variant, input_h, input_w, num_queries, num_classes, has_masks, mean, std, color_order}`. Removes the need for fragile shape-based variant autodetect. Built by `build_engine.py` next to the `.engine`. We still keep a shape-based fallback in `variant.cpp` for engines built externally.
- **Two task classes**, not one with a task enum. RF-DETR has fewer heads than YOLO; type-encoded dispatch is clearer.
- **Both prebuilt and runtime ONNX→engine flows.** Prebuilt is the default for production; runtime is a convenience for one-off use.
- **C ABI** (`include/rfdetr/c_api.h`) modeled on `mudler/rf-detr.cpp`'s `include/rfdetr.h` for FFI bindings.

---

## 5. rf-detr-cpp vs YOLOs-CPP-TensorRT — comparison

| Dimension | YOLOs-CPP-TensorRT | rf-detr-cpp | Same / Different |
|---|---|---|---|
| Distribution shape | Header-only task layer + CLI binaries | Header-only task layer + PIMPL impl + CLI binaries | **Same** |
| Backend | TensorRT 10+ via `enqueueV3` | TensorRT 10+ via `enqueueV3` | **Same** |
| Tasks supported | 5 (det, seg, pose, obb, cls) | 2 (det, seg); pose later | **Different** (narrower scope) |
| Public class shape | One class per task in `yolos::<task>` | One class per task in `rfdetr::tasks` | **Same** |
| Variant selection | Shape-based autodetect at engine-load | Sidecar JSON, shape-based fallback | **Different** (more deterministic) |
| Engine flow | Prebuilt only; trtexec script in `trt-files/` | Prebuilt **+** optional runtime ONNX→engine | **Different** (added convenience) |
| Preprocessing | Letterbox + BGR→RGB + `/255` + HWC→NCHW (single fused CUDA kernel) | **Square resize** (default) + ImageNet mean/std + BGR→RGB + HWC→NCHW (single fused CUDA kernel); optional letterbox kernel as a build/runtime option | **Different** (no letterbox by default; mean/std normalize) |
| Postprocess | Per-version anchor decode + NMS (det); proto×coef + NMS (seg); ... | Sigmoid → top-K → cxcywh→xyxy → score-thresh; **no NMS**. Mask2Former-style mask gather + bilinear upsample + binarize for seg. | **Different** (DETR head, no NMS, no anchors) |
| Pinned host I/O | Yes (`cudaHostAlloc`) | Yes | **Same** |
| Stream overlap | Single `cudaStream_t` | Single `cudaStream_t`; optional 2-stream double-buffer in batch path | **Same**, with optional extension |
| CUDA Graph capture | Yes, fixed-shape only | Yes, fixed-shape only (default for RF-DETR) | **Same** |
| Dynamic batch | Optimization profile | Optimization profile | **Same** |
| Precision | FP32 / FP16 / INT8 (Python calibration) | FP32 / FP16 / INT8 (Python calibration); TF32 toggle on Ampere+ | **Same** + minor extension |
| INT8 calibrator | EntropyCalibrator2 via Python script | EntropyCalibrator2 via Python script (writes `.cache`) | **Same** |
| Batching | Batched inference + dedicated batch CLI | Batched inference + dedicated batch CLI | **Same** |
| Examples | image / video / camera / batch | image / video / camera / batch | **Same** |
| Benchmark harness | `yolo_unified_benchmark` (quick / comprehensive) | `rfdetr_bench` + scripts (latency, throughput, mAP, mask AP) | **Same shape**, RF-DETR-specific metrics |
| Tests | Shell-driven Python-vs-C++ parity | GoogleTest unit + Python-vs-C++ parity (boxes IoU, mask IoU) | **Same shape** |
| Docker | `Dockerfile.tensorrt`, `Dockerfile.tensorrt.jetson` | `Dockerfile.x86_64`, `Dockerfile.jetson` | **Same** |
| Jetson support | Orin (CC 87) | Orin NX (CC 87) **and** AGX Thor (CC 101) | **Different** (Thor explicit) |
| C ABI | Not exposed | Stable opaque-handle C ABI | **Different** (added) |
| ROS support | No | Optional ROS 2 node in a separate sibling repo (later) | **Different** (added later) |
| License | MIT | Apache-2.0 (matches upstream RF-DETR) | **Different** |

---

## 6. Step-by-step build plan

Incremental milestones. Each step ends with something runnable, and we don't move on until the prior step's verifications pass on at least one device.

**Step 0 — Project skeleton (no model yet)**
- Empty repo skeleton matching §4. Top-level `CMakeLists.txt`, `LICENSE`, `.clang-format`, `.clang-tidy`, GH Actions stub, README scaffold.
- CMake declares CXX + CUDA, finds TensorRT, OpenCV, CUDA. Options: `RFDETR_BUILD_APPS`, `RFDETR_BUILD_BENCHMARKS`, `RFDETR_BUILD_TESTS`, `RFDETR_BUILD_C_API`, `RFDETR_USE_OPENCV_CUDA`, `RFDETR_CUDA_ARCH`.
- `core/trt_logger`, `core/cuda_check`, `core/types`, `core/variant` (enum + meta table), `core/engine_meta` (sidecar JSON parser).
- A "hello, deserialize" tool that just loads an `.engine` and prints binding shapes.

**Step 1 — Engine export + build scripts (Python)**
- `trt-files/scripts/export_onnx.py`: wraps `rfdetr.RFDETRBase().export(format="onnx")` with sane defaults; writes `<name>.onnx` and `<name>.json` sidecar with `(variant, input_hw, num_queries, num_classes, has_masks, mean, std, color_order)`.
- `trt-files/scripts/build_engine.py`: `trtexec` wrapper, FP32/FP16, optimization profile, copies sidecar next to `.engine`.
- `trt-files/scripts/verify_engine.py`: deserializes engine, prints binding shapes/dtypes; sanity-check fixture for Step 2.

**Step 2 — `TrtSession` (no preprocess, no postprocess yet)**
- `core/trt_session.{hpp,cpp}`: load engine, parse bindings, allocate device buffers + pinned host buffers, owns `cudaStream_t`. Methods: `setInput(host_ptr, bytes)`, `infer()`, `getOutput(idx, host_ptr, bytes)`. Sync-copy first; async + graph capture later.
- Unit test: feed a numpy-saved fixture input → run `infer()` → bit-exact compare outputs against a Python ORT reference within FP32 tolerance.

**Step 3 — CUDA preprocess kernel**
- `core/cuda_preprocess.cu` `squareResizeNormalizeKernel(uint8_t* src_bgr, int sH, int sW, float* dst_chw, int R, float3 mean, float3 inv_std, bool swap_to_rgb)`. Fused: bilinear resize → optional BGR→RGB → `pixel/255` → `(x − μ)/σ` → planar CHW write.
- Pinned-host upload of source `cv::Mat`, `cudaMemcpyAsync`, kernel, `cudaStreamSynchronize` before infer.
- Numerical-parity test against Python `F.resize → F.normalize` with tolerance ≤ 1e-3.

**Step 4 — Detection postprocess (CPU first, GPU later)**
- `core/cuda_postprocess.cu` *optional path*: GPU sigmoid + flat top-K via `cub::DeviceRadixSort` (skip if it's not on the critical path).
- CPU fallback in `tasks/detector.cpp`: read `dets`/`labels` via pinned-host async D2H; sigmoid + flatten + `std::partial_sort` top-K + cxcywh→xyxy × `[W,H,W,H]` + +1 class offset + score threshold filter. Lift the decode shape from `Ar-Ray-code/rf_detr_ros_cpp` (most cited and correct) and `zz990099/rf_detr_cpp`.

**Step 5 — `RFDetrDetector` end-to-end (image)**
- `tasks/detector.{hpp,cpp}` (PIMPL) wrapping `TrtSession` + preprocess + postprocess. Public methods: `Detector(engine_path, options)`, `detect(cv::Mat) → std::vector<Detection>`, `detect_batch(span<cv::Mat>)`.
- `apps/rfdetr_image.cpp`: load engine, run on a JPEG, draw boxes via `core/drawing`.
- Verify against Python `predict()` on a fixed image set: mAP IoU≥0.5 with the Python reference (we expect ≥99% per-detection class+box agreement).

**Step 6 — Async stream overlap + CUDA Graph capture**
- Switch `TrtSession` to async H2D / `enqueueV3` / async D2H on the same stream. Add CUDA Graph capture path gated by `engine.has_dynamic_shapes() == false`. Warm-up 10 iterations before capture/timing.
- Re-run latency benchmark on RTX 5070 Ti — expect a measurable drop vs Step 5.

**Step 7 — Video and camera apps**
- `apps/rfdetr_video.cpp`, `apps/rfdetr_camera.cpp`. Sequential pipeline first; double-buffered (capture stream / infer stream) variant gated by `--pipelined`.

**Step 8 — Batch + dynamic shape**
- `build_engine.py` supports `--profiles min:opt:max` for batch (and optionally resolution).
- `apps/rfdetr_batch.cpp`: directory of images → batched inference. Profile-aware `setInputShape()` per call.

**Step 9 — INT8 calibration**
- `trt-files/scripts/int8_calibrate.py`: TRT IInt8EntropyCalibrator2, reads a user-supplied calibration directory, writes `<engine>.calib`.
- `build_engine.py --int8 --calib <dir>` produces an INT8 `.engine`.
- Add an accuracy guard: refuse to ship the INT8 engine if mAP delta vs FP16 exceeds a configured threshold.

**Step 10 — Segmenter (Mask2Former-style)**
- `tasks/segmenter.{hpp,cpp}`. After detection top-K, gather `masks[query_idx]` shape `(K, R/4, R/4)`. CUDA bilinear upsample to `(H_orig, W_orig)`. Threshold at 0 → binary mask. Return `Detection + cv::Mat mask` (or polygon contour, optional).
- `apps/rfdetr_image.cpp` gains `--seg` mode.
- Cross-check against `marcoslucianops/DeepStream-Yolo-Seg`'s seg parser and `mudler/rf-detr.cpp`'s `src/postprocess.cpp` for layout sanity.

**Step 11 — C ABI + minimal example consumer**
- `include/rfdetr/c_api.h` + `src/c_api.cpp`. Mirror `mudler/rf-detr.cpp`'s `include/rfdetr.h` shape: opaque context, `init/detect/free`, mask buffers, log callback, `image_size`/`num_queries`/`num_classes` getters.
- `examples/minimal_detector/` consumes the installed package via `find_package(rfdetr)`.

**Step 12 — Docker + Jetson**
- `Dockerfile.x86_64` (CUDA 12.x + TRT 10 + Ubuntu 22.04) with a `--target prod` and `--target dev` split.
- `Dockerfile.jetson` for JetPack 6.x; `CMAKE_CUDA_ARCHITECTURES=87;101` (Orin / Thor).
- Native-build instructions in `docs/jetson.md`.

**Step 13 — Benchmark harness**
- `apps/rfdetr_bench.cpp` + `benchmarks/scripts/`: latency (mean / p50 / p99), throughput (img/s, frames/s, batched), warm-up policy, JSON output tagged with device/precision/variant. See §7.

**Step 14 — Polish**
- README, docs/, GH releases with prebuilt engines per device-precision pair (optional), CI matrix.

---

## 7. Benchmarking & validation

We have to validate two orthogonal axes per build: **correctness** (does the C++ path agree with the Python reference?) and **performance** (latency / throughput) across three devices and three precisions.

### 7.1 Devices

| Device | GPU | CC | TRT | JetPack | Notes |
|---|---|---|---|---|---|
| Workstation | NVIDIA RTX 5070 Ti (Blackwell) | 12.0 | TRT 10.x for Blackwell | n/a | Discrete, lots of headroom |
| Jetson AGX Thor | Thor SoC | 10.1 (Blackwell-class Jetson) | JetPack 7.x bundled TRT | 7.x | New silicon — confirm CC at first build |
| Jetson Orin NX 16 GB | Orin SoC (Ampere) | 8.7 | JetPack 6.x bundled TRT | 6.x | Power modes matter; pin to `MAXN` |

Pin clocks before timing: `nvidia-smi -lgc <max>,<max>` on desktop; `jetson_clocks` (and `nvpmodel -m 0`) on Jetson.

### 7.2 Test inputs

- **Correctness fixture** (`tests/data/`): 20 hand-picked COCO val2017 images covering small/medium/large targets, dense scenes, and aspect ratios far from 1:1. Plus a synthetic gradient image for kernel-level sanity.
- **Benchmark fixture** (`benchmarks/coco_subset/`): 100 fixed COCO val2017 images chosen to span aspect ratios and instance counts. Stored as a manifest of image IDs; downloaded by `scripts/download_coco_subset.sh`. `annotations.json` is a slice of `instances_val2017.json` for the same image IDs.

### 7.3 Variant × precision matrix

Per device, run:

| Variant | Precision | Batch | Notes |
|---|---|---|---|
| Nano (384) | FP16, INT8 | 1, 8 | latency-critical |
| Small (512) | FP16, INT8 | 1, 8 | |
| Base (560) | FP16, INT8 | 1, 4 | |
| Medium (576) | FP16, INT8 | 1, 4 | |
| Large (704) | FP16, INT8 | 1 | throughput-critical |
| Seg-Small (384) | FP16, INT8 | 1, 4 | |
| Seg-Medium (432) | FP16, INT8 | 1, 4 | |
| Seg-Large (504) | FP16, INT8 | 1 | |

FP32 is included on RTX 5070 Ti only (sanity-check baseline). Skip FP32 on Jetson (memory-bound). On Orin NX, drop the largest variants if memory-tight — log skip reasons.

### 7.4 Latency methodology

Per (device, variant, precision, batch):

1. **Warm-up**: 50 iterations on the same input. Discard.
2. **Measure**: 500 iterations, single fixed input, measure end-to-end *including* H2D + kernel-prep + `enqueueV3` (or `cudaGraphLaunch`) + D2H + postprocess. Use `cudaEventRecord` around the GPU-side block; `std::chrono::steady_clock` around the full call.
3. **Report**: mean / p50 / p90 / p99 / min / max in ms. Also report **GPU-only** time (event diff) vs **end-to-end** wall time so we can see how much postprocess + H2D costs.
4. Repeat without CUDA Graph capture and with — confirm the gain.

### 7.5 Throughput methodology

Per (device, variant, precision, batch):

1. Warm-up 50 iters.
2. Run 1000 iters on rotating images (cycle through the 100-image fixture so caches don't keep us honest). Single producer thread feeds, single inference thread consumes.
3. Report: total wall time, **frames per second** = 1000 × batch / wall_time, GPU utilization (`nvidia-smi dmon` or `tegrastats` log captured for the run).
4. Pipelined variant: producer-decode + infer overlapped on two streams. Report separately.

### 7.6 Accuracy validation

Two independent checks per (variant, precision):

**(A) Per-image parity vs Python reference.** Run the same fixed 100-image fixture through the Python `rfdetr` `predict()` and our C++ `Detector`/`Segmenter`. For each image:
- Boxes: greedy IoU matching at IoU ≥ 0.7. Pass = ≥99% of FP16 detections matched, mean class-id agreement = 100% on matched.
- Masks: matched detections must have mask IoU ≥ 0.95 vs Python.
- We expect FP32 to be near bit-exact on box coords (within float rounding), FP16 within a few px, INT8 measurably degraded but bounded.

**(B) COCO-style mAP** via `pycocotools` on the 100-image subset (full val2017 takes too long on Jetson).
- For each (variant, precision): mAP\@0.5:0.95, mAP\@0.5, mAP\@0.75. For seg: same plus mask-mAP.
- INT8 ship-gate: mAP\@0.5:0.95 must not drop more than 1.0 absolute point vs FP16. If it does, the calibration set is bad — re-run with a wider/larger one.

Both checks live in `tests/parity/` and run in CI on a CPU-only mock backend (skipping GPU asserts) plus on a GPU runner when available.

### 7.7 Benchmark output and comparison

`apps/rfdetr_bench --device <tag> --variant <v> --precision <p> --batch <b> --report results/<tag>_<v>_<p>_b<b>.json`

JSON schema:
```json
{
  "device": "rtx-5070ti", "tegrastats": null,
  "variant": "small", "input_hw": [512, 512],
  "precision": "fp16", "batch": 1,
  "trt_version": "10.x.y", "cuda_version": "12.x", "driver": "...",
  "warmup_iters": 50, "measure_iters": 500,
  "latency_ms": { "mean": ..., "p50": ..., "p90": ..., "p99": ..., "min": ..., "max": ... },
  "gpu_time_ms": { "mean": ..., "p99": ... },
  "throughput_fps": ...,
  "cuda_graph": true, "pipelined": false,
  "map_50_95": ..., "map_50": ..., "map_75": ..., "mask_map_50_95": ...
}
```

`benchmarks/scripts/compare_devices.py` ingests `results/*.json` and produces a single Markdown table of (device × variant × precision × batch) → (latency p50, latency p99, FPS, mAP). This is the artifact we paste into release notes.

### 7.8 Things to watch on each device

- **RTX 5070 Ti (Blackwell)**: confirm TRT version supports SM 12.0; build with `-DCMAKE_CUDA_ARCHITECTURES=120` (or whatever SM nvcc reports). FP8 is available in hardware — out of scope for v1, note it as future work.
- **AGX Thor**: new silicon. First task on this device is to confirm CC and TRT version (`/usr/src/tensorrt/bin/trtexec --version`). Build with the right `CMAKE_CUDA_ARCHITECTURES` (likely 101). Pin power mode to MAXN.
- **Orin NX 16 GB**: 16 GB shared memory means the largest variants (Large 704, Seg-XL 624, Seg-2XL 768) may OOM at batch 1 with FP16. Track failures, prefer INT8 at the top of the size range. Always run with `nvpmodel -m 0 && jetson_clocks` and capture `tegrastats` alongside the run.
- **All Jetson**: DLA cores exist on Orin/Thor but are not part of the v1 plan. Note as a follow-up.
