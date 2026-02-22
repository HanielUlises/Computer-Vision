# Model Interchange / ONNX

High-performance ONNX Runtime inference wrapper for the **hpcv** computer
vision library. Designed to slot into the existing repo under:

```
Learned Representations/
└── model_interchange/
    └── onnx/          ← this directory
```

---

## Features

| | |
|---|---|
| **Backends** | CPU · CUDA · TensorRT (auto-fallback) · OpenVINO (auto-fallback) |
| **Tasks** | Object detection (YOLOv8-style output) — easily extended to classification / segmentation |
| **Preprocessing** | Letterbox resize · batch NCHW blob · ImageNet / [0,1] / [-1,1] normalisation |
| **Postprocessing** | Confidence filter · OpenCV NMS · unletterbox back to original coords |
| **API** | Raw `run()` for arbitrary tensors + high-level `detect()` / `detectBatch()` |

---

## Directory layout

```
onnx/
├── include/
│   ├── onnx_inference_engine.hpp   ← main engine class
│   └── preprocessor.hpp            ← image preprocessing utilities
├── src/
│   ├── onnx_inference_engine.cpp
│   ├── preprocessor.cpp
│   └── main.cpp                    ← CLI demo (image / webcam / video)
├── tests/
│   └── test_onnx_engine.cpp        ← Catch2 unit tests
└── CMakeLists.txt
```

---

## Dependencies

| Library | Version | Notes |
|---|---|---|
| ONNX Runtime | ≥ 1.16 | With CUDA/TRT providers for GPU |
| OpenCV | ≥ 4.5 | core · imgproc · dnn · highgui |
| CUDA Toolkit | ≥ 11.8 | Only for GPU backends |
| Catch2 | ≥ 3.x | Optional, for tests |

### Installing ONNX Runtime (Ubuntu)
```bash
# Download the GPU release from https://github.com/microsoft/onnxruntime/releases!!
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/\
onnxruntime-linux-x64-gpu-1.18.1.tgz
tar xzf onnxruntime-linux-x64-gpu-1.18.1.tgz
export ONNXRUNTIME_ROOT=$PWD/onnxruntime-linux-x64-gpu-1.18.1
```

---

## Build

```bash
cd Learned\ Representations/Model\ Interchange/ONNX
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=$ONNXRUNTIME_ROOT

cmake --build . -j$(nproc)
```

To skip tests:
```bash
cmake .. -DBUILD_TESTS=OFF
```

---

## Usage

### CLI demo
```bash
# Run on an image
./onnx_demo --model yolov8n.onnx --source photo.jpg --labels coco.txt

# Webcam (device 0)
./onnx_demo --model yolov8n.onnx --source 0

# CPU only
./onnx_demo --model yolov8n.onnx --source photo.jpg --cpu

# Confidence / NMS thresholds
./onnx_demo --model yolov8n.onnx --source photo.jpg --conf 0.4 --nms 0.5
```

### API (C++)
```cpp
#include "OnnxInferenceEngine.hpp"

hpcv::OnnxInferenceEngine::Config cfg;
cfg.modelPath = "yolov8n.onnx";
cfg.provider  = hpcv::ExecutionProvider::CUDA;   // or CPU / TensorRT
cfg.deviceId  = 0;

hpcv::OnnxInferenceEngine engine(cfg);
engine.loadLabels("coco.txt");

cv::Mat frame = cv::imread("photo.jpg");
auto dets = engine.detect(frame, 0.5f, 0.45f);

for (auto& d : dets)
    std::cout << d.label << "  conf=" << d.confidence
              << "  bbox=(" << d.bbox.x << "," << d.bbox.y
              << "," << d.bbox.width << "," << d.bbox.height << ")\n";

auto results = engine.detectBatch({frame1, frame2, frame3}, 0.5f, 0.45f);
```

### Extending to classification

Override the output decoder — the raw `run()` API gives you the output
tensors directly:

```cpp
auto outputs   = engine.run(inputs);            // raw OrtValue vector
const float* logits = outputs[0].GetTensorData<float>();
int cls = std::max_element(logits, logits + numClasses) - logits;
```

---

## Exporting a YOLOv8 model

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx opset=17 simplify=True
```

---

## Running tests

```bash
# Integration test requires a model path
export HPCV_TEST_MODEL=yolov8n.onnx
cd build && ctest --output-on-failure
```