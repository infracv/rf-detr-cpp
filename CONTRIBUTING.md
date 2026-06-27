# Contributing to RF-DETR C++

Thanks for your interest in improving RF-DETR C++. This document explains how to set up your environment, the standards we follow, and how to submit changes.

By participating in this project, you agree to abide by our [Code of Conduct](./CODE_OF_CONDUCT.md).

---

## Ways to Contribute

- Report issues with reproducible test cases
- Improve documentation and examples
- Fix bugs or improve performance
- Add new precision modes, model variants, or platform support
- Help expand benchmark coverage

For non-trivial changes (new APIs, architectural refactors, plugin systems) please open an issue first to discuss the approach before writing code.

---

## Reporting Issues

Before opening an issue, please check existing issues to avoid duplicates.

A good bug report includes:

- **GPU model** and CUDA Compute Capability (e.g. RTX 5070 Ti, sm_120)
- **CUDA Toolkit version** (`nvcc --version`)
- **TensorRT version** (`dpkg -l | grep tensorrt` or tarball version)
- **Driver version** (`nvidia-smi`)
- **OS and compiler** (e.g. Ubuntu 24.04, GCC 13.3)
- **Build command used** (full CMake invocation)
- **Minimal reproducer** — a short command sequence or code snippet that triggers the bug
- **Expected vs actual behavior**
- **Full error output** (not a screenshot — paste the text)

For performance regressions, include before/after numbers from `rfdetr_bench`.

---

## Development Setup

### Prerequisites

| Tool | Minimum | Tested |
|:-----|:-------:|:------:|
| CUDA Toolkit | 12.0 | 12.8 |
| TensorRT | 10.0 | 11.x |
| CMake | 3.20 | 3.28 |
| GCC | 9 | 13.3 |
| Clang | 10 | 16 |
| OpenCV | 4.5 | 4.6 |

### Fork and Clone

```sh
git clone https://github.com/<your-username>/rf-detr-cpp.git
cd rf-detr-cpp
git remote add upstream https://github.com/infracv/rf-detr-cpp.git
```

### Build

```sh
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j$(nproc)
```

Replace `120` with your GPU's compute capability. See the README for the full GPU-to-arch mapping.

### Debug Builds

```sh
cmake -B build-debug -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-debug -j$(nproc)
```

For memory and kernel correctness checks, use NVIDIA Compute Sanitizer:

```sh
compute-sanitizer --tool memcheck ./build-debug/rfdetr_smoke <engine>
compute-sanitizer --tool racecheck ./build-debug/rfdetr_smoke <engine>
```

---

## Coding Standards

### C++ Style

- Follow the project `.clang-format` (run `clang-format -i <file>` before committing).
- Run `clang-tidy` for new code where practical.
- 4-space indentation, no tabs.
- C++17 only — do not introduce C++20/23 features.
- RAII for all resources. Use `std::unique_ptr` with custom deleters for CUDA and TensorRT handles.
- `[[nodiscard]]` on factory functions and observable accessors.
- Public headers must compile without CUDA or TensorRT headers exposed — use PIMPL.
- No `using namespace` in headers.
- Prefer `std::filesystem::path` over raw strings for file paths.

### CUDA Style

- Wrap every CUDA API call with `RFDETR_CUDA_CHECK(...)`.
- Follow kernel launches with an error check.
- Document stream and buffer ownership in function comments.
- Avoid implicit host-device synchronization in hot paths.
- Use `__restrict__` on non-aliasing pointer parameters in kernels.

### TensorRT Style

- Use `nvinfer1::IExecutionContext::enqueueV3` (the V2 variant is deprecated).
- Smart-pointer all TRT handles (`IRuntime`, `ICudaEngine`, `IExecutionContext`, `IBuilder`).
- Engine plan files are GPU-, driver-, and TRT-version-specific. Do not assume portability.

### Comments

- Explain **why**, not **what**.
- Do not leave commented-out code, dated TODOs, or "Step 1/2/3" development notes.
- Public headers may use lightweight Doxygen (`///` or `/** */`) for API documentation.

---

## Testing

Until a full test suite lands, every change must at minimum:

1. Build cleanly with no new warnings (`-Wall -Wextra`).
2. Pass the smoke test:
   ```sh
   ./build/rfdetr_smoke trt-files/onnx/rf-detr-nano-fp16.engine
   ```
3. Run successfully against the example image:
   ```sh
   ./build/examples/example_image_det trt-files/onnx/rf-detr-nano-fp16.engine asset/test_img.jpg
   ```

For changes touching CUDA kernels, also run `compute-sanitizer` (see Development Setup).

For performance-sensitive changes, include before/after `rfdetr_bench` numbers in the PR description.

---

## Documentation

- Update the README if you change the public API, CLI flags, or build options.
- Update inline comments and Doxygen annotations on touched public APIs.
- Add an entry to the examples directory if you introduce a new user-facing pattern.

---

## Commit Guidelines

### Commit Messages

- Keep the subject line under 72 characters.
- Use the imperative mood ("Add INT8 calibrator" not "Added INT8 calibrator").
- Describe **why** in the body when the change is non-obvious.
- Reference issues with `Closes #123` or `Refs #123`.

### Sign-off (DCO)

This project uses the [Developer Certificate of Origin](https://developercertificate.org). Every commit must be signed off:

```sh
git commit -s -m "Your message"
```

This appends a `Signed-off-by:` line certifying that you wrote the code (or have the right to submit it) and that you understand it will be published under the project's license.

If you forget, amend the last commit:

```sh
git commit --amend -s --no-edit
```

### License Headers

Add the SPDX identifier to new source files:

```cpp
// SPDX-License-Identifier: Apache-2.0
```

---

## Pull Request Process

1. Create a branch from `develop`: `git checkout -b feat/short-description`
2. Make focused, atomic commits. Split unrelated changes into separate PRs.
3. Rebase on the latest `develop` before opening the PR.
4. Open the PR against `develop`, not `main`. The `main` branch tracks released versions.
5. Fill in the PR template. Include:
   - What changed and why
   - How you tested it (commands, hardware, engine versions)
   - Any benchmark numbers for performance-relevant changes
6. Be responsive to review feedback. Force-push is fine on feature branches.

PRs are squashed or rebased on merge; either preserve a clean history.

---

## Contributor License

By contributing, you agree that your contributions will be licensed under the [Apache License 2.0](./LICENSE), the same license that covers the rest of this project.

---

## Questions

Open a GitHub Discussion or an issue tagged `question`. For private inquiries, contact the maintainer listed in the repository profile.
