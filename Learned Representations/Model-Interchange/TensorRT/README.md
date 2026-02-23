# Model Interchange / TensorRT

Native **TensorRT 8/9/10** inference backend for the **hpcv** library.  
Same namespace, same `detection` struct, same CLI flags as the ONNX folder —
just swap the library and get maximum GPU throughput.

---

## Features

| | |
|---|---|
| **Backends** | TensorRT (CUDA), optional FP16 & INT8 |
| **Build source** | Pre-built `.engine` file **or** auto-build from `.onnx` |
| **Tasks** | YOLOv8-style object detection; raw `run_raw()` for anything else |
| **Preprocessing** | Letterbox · NCHW blob · [0,1] / ImageNet / [-1,1] norms |
| **Postprocessing** | Confidence filter · OpenCV NMS · unletterbox |

---

## Dependencies

| Library | Version | Notes |
|---|---|---|
| TensorRT | ≥ 8.6 | `nvinfer`, `nvonnxparser` |
| CUDA Toolkit | ≥ 11.8 | `cudart` |
| OpenCV | ≥ 4.5 | core · imgproc · dnn · highgui |
| Catch2 | ≥ 3.x | Optional, tests only |

### Installing TensorRT (Ubuntu)
```bash
# Via NVIDIA's apt repo (recommended):
sudo apt-get install tensorrt

# OR download the tar-ball from https://developer.nvidia.com/tensorrt
# and point cmake at it with -DTENSORRT_ROOT=<path>
```

---

## Build

```bash
cd Model-Interchange/TensorRT
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTENSORRT_ROOT=/usr/local/tensorrt   # omit if TRT is in /usr

cmake --build . -j$(nproc)
```

---

## Usage

### Option A — supply a pre-built engine (fastest startup)
```bash
./trt_demo --engine yolov8n.engine --source photo.jpg --labels coco.txt
```

### Option B — build the engine from ONNX on first run
```bash
# Export ONNX first:  yolo export model=yolov8n.pt format=onnx opset=17
./trt_demo --onnx yolov8n.onnx --save-engine yolov8n.engine \
           --source photo.jpg --labels coco.txt
# Subsequent runs use the cached .engine file
./trt_demo --engine yolov8n.engine --source 0
```

### Webcam / video
```bash
./trt_demo --engine yolov8n.engine --source 0           # webcam
./trt_demo --engine yolov8n.engine --source video.mp4
```

### Disable FP16
```bash
./trt_demo --onnx yolov8n.onnx --no-fp16 --source photo.jpg
```

### C++ API
```cpp
#include "trt_inference_engine.hpp"

hpcv::trt_inference_engine::config cfg;
cfg.engine_path = "yolov8n.engine";   // or set cfg.onnx_path to build
cfg.fp16        = true;
cfg.device_id   = 0;

hpcv::trt_inference_engine engine(cfg);
engine.load_labels("coco.txt");

cv::Mat frame = cv::imread("photo.jpg");
auto dets = engine.detect(frame, 0.5f, 0.45f);

for (auto& d : dets)
    std::cout << d.label << "  " << d.confidence
              << "  (" << d.bbox.x << "," << d.bbox.y << ")\n";

// Batch inference
auto results = engine.detect_batch({frame1, frame2}, 0.5f, 0.45f);

// Raw tensor access
auto raw = engine.run_raw(blob, batch_size);
```

---

## Running tests
Two options either a prebuilt or using the onnx one (almost same)
```bash
export HPCV_TRT_ENGINE=yolov8n.engine   # pre-built engine
export HPCV_TEST_ONNX=yolov8n.onnx      # for build-from-onnx test

cd build && ctest --output-on-failure
```

---

## Notes

- Engine files are GPU-architecture specific — rebuild when changing GPU.
- FP16 requires Turing (RTX 20xx) or newer; INT8 additionally requires a calibration dataset (not wired here — subclass `IInt8Calibrator` and pass it to `IBuilderConfig`).
- For streaming / async inference, replace `executeV2` with `enqueueV3` and supply a CUDA stream.
