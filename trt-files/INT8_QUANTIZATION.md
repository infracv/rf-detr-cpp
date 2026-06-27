# INT8 Quantization

Two paths depending on your TensorRT version.

---

## Calibration Dataset

RF-DETR is trained on COCO 80 classes. NVIDIA recommends ~5,000 representative images for COCO-trained models. Labels are not used, only pixel values matter.

```sh
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip   # → val2017/ with 5,000 images
```

| Calibration images | Quality | Use case |
|:------------------:|---------|----------|
| 100–200 | Quick smoke test | CI / fast iteration |
| 1,000–2,000 | Good | Prototype |
| **5,000 (COCO val2017)** | **Best** | **Production** |

The `val2017/` folder can be reused across all model variants.

---

## TRT 11+ (QDQ Path)

```sh
# Step 1: Insert Quantize/Dequantize nodes via calibration
python trt-files/scripts/convert_int8.py \
    --onnx trt-files/onnx/rf-detr-nano.onnx \
    --images val2017/ --max-images 5000 \
    --out trt-files/onnx/rf-detr-nano-int8.onnx

# Step 2: Build engine (TRT reads the embedded QDQ nodes)
./build/rfdetr_build \
    --onnx   trt-files/onnx/rf-detr-nano-int8.onnx \
    --engine trt-files/onnx/rf-detr-nano-int8.engine \
    --precision fp32
```

`convert_int8.py` excludes `LayerNorm`, `Softmax`, and `Gelu` by default to preserve accuracy. Pass `--no-exclude-sensitive` to quantize everything.

---

## TRT < 11 (Implicit Calibration Path)

```sh
# Step 1: Generate calibration cache
python trt-files/scripts/int8_calibrate.py \
    --onnx trt-files/onnx/rf-detr-nano.onnx \
    --images val2017/ --max-images 5000 \
    --output trt-files/onnx/rf-detr-nano.calib

# Step 2: Build with C++ builder
./build/rfdetr_build \
    --onnx trt-files/onnx/rf-detr-nano.onnx --precision int8 \
    --calib trt-files/onnx/rf-detr-nano.calib
```

---

## Notes

- QDQ ONNX files and `.calib` caches are GPU-architecture-specific. Rebuild them when changing hardware.
- INT8 engines typically use 10–20% less GPU memory than FP16 at the cost of a small accuracy drop.
- If INT8 inference yields zero detections, lower your detection threshold first before assuming a calibration issue.
