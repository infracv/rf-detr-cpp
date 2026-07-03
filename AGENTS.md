# Agent Instructions

This file gives AI coding assistants (Claude Code, Codex, Cursor, Copilot, Aider, etc.) the context they need to make safe, useful contributions to this repository.

If you are an AI assistant: read this entire file before making changes. It is short by design.

---

## Project Overview

RF-DETR C++ is a TensorRT-backed inference library for the RF-DETR object detection model. The pipeline runs entirely in C++ with CUDA-accelerated preprocessing, async H2D transfers via pinned memory, and CUDA Graph capture for low-overhead dispatch. The library is consumed as a shared object (`librfdetr.so`) and ships a CLI toolset, examples, and a C ABI for FFI.

Target users are C++ developers building real-time computer vision systems on NVIDIA GPUs. Python is required only for one-time ONNX export and calibration — never at runtime.

---

## Toolchain

- **C++17** (GCC 13 / Clang 16 tested)
- **CUDA 12.0+**
- **TensorRT 10.x or 11.x** (TRT 11 uses strongly-typed networks)
- **CMake 3.20+**
- **OpenCV 4.5+** (core, imgproc, imgcodecs, videoio, highgui)

Always state the toolchain assumption when proposing code. Do not introduce C++20/23 idioms unless the user asks.

---

## Repository Layout

```
include/rfdetr/       Public headers (consumers #include these)
  core/               Postprocessing, drawing, types, log, engine meta, variant
  tasks/              High-level task APIs (detector, segmenter)
  c_api.h             C ABI for FFI bindings
src/                  Implementation — mirrors include/ structure
  internal/           Implementation-only headers (never installed)
    trt_session.hpp       TensorRT runtime/engine/context wrapper
    cuda_preprocess.cuh   CUDA preprocessing kernel declarations
    trt_logger.hpp        ILogger → rfdetr log channel adapter
    cuda_check.hpp        RFDETR_CUDA_CHECK / RFDETR_TRT_CHECK macros
    cuda_raii.hpp         DevPtr / HostPtr RAII wrappers for CUDA memory
    mask_decode.cuh       GPU mask decode kernel declaration
  core/cuda_preprocess.cu   Fused resize + normalise CUDA kernel
  core/mask_decode.cu       GPU bilinear upsample + threshold mask kernel
apps/                 CLI binaries (build, detect, seg, video, batch, bench, ...)
  CLI_TOOLS.md        CLI reference for all tools
  cli_helpers.hpp     Shared argument parsing helpers
examples/             End-user example programs with utils.hpp
  EXAMPLES.md         Build and run instructions for examples
trt-files/scripts/    Python utilities for ONNX export and INT8 calibration
benchmarks/           Benchmark scripts and results
asset/                Test images and videos
FindTensorRT.cmake    CMake module to locate TensorRT
```

Internal headers (`src/internal/`) are **not** part of the public API and must not be included by consumers of `librfdetr.so`. The paths under `include/rfdetr/core/` for those four files are thin shims retained for backward compatibility with tools in `apps/`; new code should include from `src/internal/` or `../internal/` directly.

Do not create new top-level directories without checking first.

---

## Build & Run

See `README.md` (Installation section) for the full build command and GPU arch table.

Always pass `-DCMAKE_CUDA_ARCHITECTURES` explicitly. RTX 30xx = 86, RTX 40xx = 89, RTX 50xx = 120.

CMake options: `RFDETR_BUILD_APPS`, `RFDETR_BUILD_EXAMPLES`, `RFDETR_BUILD_BENCHMARKS`, `RFDETR_BUILD_C_API`, `RFDETR_BUILD_TESTS`, `RFDETR_BUILD_SHARED`, `RFDETR_USE_OPENCV_CUDA`.

---

## Code Style

- `.clang-format` and `.clang-tidy` live at the repo root — respect them.
- 4-space indent, 100-column soft limit, braces on same line for function definitions.
- Prefer `std::unique_ptr` over raw `new`/`delete`.
- Use RAII for all CUDA and TRT handles — never call `cudaFree`, `destroy()`, or `delete` manually outside a deleter.
- `[[nodiscard]]` on factory functions and accessors.
- No `using namespace std;` in headers. No `using namespace rfdetr;` anywhere.
- Public headers must remain TensorRT/CUDA-free — implementation details belong behind PIMPL (see `RFDetrDetector::Impl`).
- Keep comments terse. Explain **why**, not **what**. Do not narrate the obvious.

---

## CUDA / TensorRT Rules

- Every CUDA call goes through `RFDETR_CUDA_CHECK(...)` (see `src/internal/cuda_check.hpp`).
- After every kernel launch, follow with `cudaPeekAtLastError()` if not already inside a `_CHECK` macro.
- Do not silently swallow `cudaError_t` — bubble it up via exceptions or the check macro.
- Stream ownership: the `TrtSession` owns its stream; callers must never destroy it.
- Engine plan files (`.engine`) are GPU-, driver-, and TRT-version-specific. Never assume cross-compatibility.
- TRT 11 has no `kFP16` builder flag — FP16 is achieved by pre-converting ONNX weights (handled automatically by `rfdetr_build` via `convert_fp16.py`).
- INT8 path on TRT 11+ uses **QDQ ONNX** via `convert_int8.py`, not implicit calibration.

---

## Memory & Resource Ownership

- `RFDetrDetector` and `RFDetrSegmenter` each own a `TrtSession`, which owns the `IRuntime`, `ICudaEngine`, `IExecutionContext`, and `cudaStream_t`.
- Device and pinned-host buffers in `TrtSession` are managed via `DevPtr` / `HostPtr` RAII wrappers (see `src/internal/cuda_raii.hpp`). Never call `cudaFree` / `cudaFreeHost` manually outside a deleter.
- `ImagePreprocessor` device and staging buffers are also `DevPtr` / `HostPtr` — destructor is `= default`.
- CUDA Graph captures (`cudaGraph_t`, `cudaGraphExec_t`) are owned by `Impl` structs and destroyed in `destroy_graph_()`.
- GPU mask decode (`src/core/mask_decode.cu`) allocates its own per-call scratch buffers via RAII and frees them automatically before returning.
- The C ABI in `src/c_api.cpp` wraps the C++ API — never throw exceptions across the boundary.

---

## Pre-commit Checklist

Before pushing or opening a PR, run:

```sh
# 1. Build cleanly
cmake --build build -j$(nproc)

# 2. Smoke-test the detector
./build/rfdetr_smoke trt-files/onnx/rf-detr-nano-fp16.engine

# 3. Run examples
./build/examples/example_image_det trt-files/onnx/rf-detr-nano-fp16.engine asset/test_img.jpg

# 4. Smoke-test segmentation (if a seg engine is available)
./build/rfdetr_seg --engine trt-files/onnx/rf-detr-seg-nano-fp16.engine \
    --image asset/test_img.jpg --out /tmp/seg_result.jpg

# 5. (Optional) compute-sanitizer for CUDA changes
compute-sanitizer --tool memcheck ./build/rfdetr_smoke <engine>
```

---

## Things Not To Do

- Do not modify files under `build/`, `out/`, or `trt-files/onnx/` — they are generated.
- Do not add Python dependencies to the runtime. Python is allowed only in `trt-files/scripts/`.
- Do not introduce exceptions across the C ABI boundary.
- Do not include headers from `src/internal/` in public headers under `include/`. Internal types (`TrtSession`, `BindingInfo`, `nvinfer1::*`) must stay behind PIMPL.
- Do not call `cudaFree` / `cudaFreeHost` / `cudaMalloc` directly — always use `DevPtr` / `HostPtr` from `src/internal/cuda_raii.hpp` or `RFDETR_CUDA_CHECK` where raw allocation is unavoidable.
- Do not bump `CMAKE_CUDA_ARCHITECTURES` defaults without considering Jetson (87), Thor (101), and Blackwell (120).
- Do not add commented-out code, `TODO` blocks without owners, or stale "Step 1/2/3" development notes.
- Do not commit large binaries (videos, models). The `.gitignore` covers common cases — check before adding.

---

## When You Are Unsure

Ask the maintainer. Open an issue first for non-trivial architecture changes. Small bug fixes and documentation improvements can go straight to a PR.
