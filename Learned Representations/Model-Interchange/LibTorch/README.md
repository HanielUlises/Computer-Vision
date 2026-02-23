# Model Interchange / LibTorch

**PyTorch C++ (LibTorch)** inference backend for the **hpcv** library.  
Loads any model exported with `torch.jit.script()` or `torch.jit.trace()`.  
Same `hpcv` namespace, same `detection` struct and CLI flags as the ONNX and TensorRT folders.

---

## Features

| | |
|---|---|
| **Backends** | CPU, CUDA; optional FP16 on CUDA |
| **Model format** | TorchScript (`.pt` / `.torchscript`) |
| **Tasks** | YOLOv8-style object detection; raw `run()` for anything else |
| **Preprocessing** | Letterbox · NCHW `torch::Tensor` · [0,1] / ImageNet / [-1,1] norms |
| **Postprocessing** | Confidence filter · OpenCV NMS · unletterbox |

---

## Dependencies

| Library | Version | Notes |
|---|---|---|
| LibTorch | ≥ 2.0 | GPU build recommended |
| OpenCV | ≥ 4.5 | core · imgproc · dnn · highgui |
| CUDA Toolkit | ≥ 11.8 | Optional (CPU-only build works too) |
| Catch2 | ≥ 3.x | Optional, tests only |

### Download LibTorch
```bash
# Visit https://pytorch.org/get-started/locally/
# Select: LibTorch | C++/Java | CUDA 12.x | Linux

wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip
unzip libtorch-*.zip
```

---

## Exporting a PyTorch model to TorchScript

```python
import torch
from ultralytics import YOLO

# Option A — trace (fastest for models without control flow)
model = YOLO("yolov8n.pt")
dummy = torch.zeros(1, 3, 640, 640)
traced = torch.jit.trace(model.model, dummy)
traced.save("yolov8n.torchscript")

# Option B — ultralytics export shortcut
model.export(format="torchscript")
```

---

## Build

```bash
cd Model-Interchange/LibTorch
mkdir build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch 
  
cmake --build . -j$(nproc)
```

If LibTorch is already on your system (e.g. installed via pip):
```bash
TORCH_ROOT=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake .. -DCMAKE_PREFIX_PATH=$TORCH_ROOT
```

---

## Usage

### CLI demo

```bash
# Image
./torch_demo --model yolov8n.torchscript --source photo.jpg --labels coco.txt

# Webcam
./torch_demo --model yolov8n.torchscript --source 0

# CPU only
./torch_demo --model yolov8n.torchscript --source photo.jpg --cpu

# FP16 on GPU
./torch_demo --model yolov8n.torchscript --source 0 --fp16

# Thresholds
./torch_demo --model yolov8n.torchscript --source photo.jpg --conf 0.4 --nms 0.5
```

### C++ API

```cpp
#include "torch_inference_engine.hpp"

hpcv::torch_inference_engine::config cfg;
cfg.model_path = "yolov8n.torchscript";
cfg.device     = hpcv::torch_inference_engine::config::device_type::cuda;
cfg.device_id  = 0;
cfg.fp16       = false;   // set true for Ampere+

hpcv::torch_inference_engine engine(cfg);
engine.load_labels("coco.txt");

cv::Mat frame = cv::imread("photo.jpg");

// Single image
auto dets = engine.detect(frame, 0.5f, 0.45f);
for (auto& d : dets)
    std::cout << d.label << "  " << d.confidence << "\n";

// Batch
auto results = engine.detect_batch({frame1, frame2}, 0.5f, 0.45f);

// Raw TorchScript forward pass
auto input  = hpcv::preprocessor::to_batch_tensor({frame}, hpcv::norm_zero_one());
auto output = engine.run(input);           // vector<torch::jit::IValue>
auto tensor = output[0].toTensor();        // access raw output
```

---

## For testing:

```bash
export HPCV_TORCH_MODEL=yolov8n.torchscript

cd build && ctest --output-on-failure
```

---

## Notes

- FP16 inference requires Ampere (RTX 30xx) or newer for best speedup.
- LibTorch shared libraries must be on `LD_LIBRARY_PATH` at runtime (the CMake install step handles rpath automatically).
- For async/streamed inference, wrap `engine.run()` calls with `torch::NoGradGuard` and CUDA streams; LibTorch is thread-safe per-module.
