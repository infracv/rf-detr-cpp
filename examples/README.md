# RF-DETR C++ Examples

Minimal, self-contained examples showing how to integrate the RF-DETR C++ library
into your own project. These are intentionally simple — no CLI argument parsing,
just the essential API calls.

## Examples

| Binary | Input | Description |
|--------|-------|-------------|
| `example_image_det` | Image file | Detect on a single image, save annotated result |
| `example_video_det` | Video file | Detect on every frame, write annotated video |
| `example_camera_det` | Camera / RTSP | Live detection with FPS overlay; `s` to snapshot, `q` to quit |

## Build

```sh
cd examples
cmake -B build -S . -DTENSORRT_DIR=/path/to/TensorRT
cmake --build build -j$(nproc)
```

> If TensorRT is installed system-wide (e.g. via apt), omit `-DTENSORRT_DIR`.

## Run

```sh
# Image
./run_image_det.sh rf-detr-nano-fp16.engine dog.jpg 0.5

# Video
./run_video_det.sh rf-detr-nano-fp16.engine video.mp4 0.5

# Camera (device 0)
./run_camera_det.sh rf-detr-nano-fp16.engine 0 0.5
```

Or run the binaries directly from `build/`:

```sh
./build/example_image_det  rf-detr-nano-fp16.engine dog.jpg 0.5
./build/example_video_det  rf-detr-nano-fp16.engine video.mp4
./build/example_camera_det rf-detr-nano-fp16.engine 0
```

## Output

Annotated results are written to `out/` in the working directory.

## Notes

- Engine files must be built first with `rfdetr_build`. See the [main README](../README.md#-model-conversion) for conversion instructions.
- TensorRT engines are GPU-architecture specific — rebuild on each target device.
- Camera example requires a display (X11 or Wayland). For headless use, remove the `cv::imshow` call and keep the `writer`.
