#pragma once

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace hpcv {


struct detection {
    cv::Rect2f  bbox;        // normalised [0,1] coords  (x,y,w,h)
    float       confidence;
    int         class_id;
    std::string label;
};


class trt_logger : public nvinfer1::ILogger {
public:
    explicit trt_logger(nvinfer1::ILogger::Severity min_sev =
                            nvinfer1::ILogger::Severity::kWARNING)
        : min_sev_(min_sev) {}

    void log(Severity sev, const char* msg) noexcept override;

private:
    Severity min_sev_;
};


class trt_inference_engine {
public:
    struct config {
        // EITHER a pre-built .engine file OR an .onnx file to build from.
        std::string engine_path;      // path to serialised .engine  (preferred)
        std::string onnx_path;        // path to .onnx  (used when engine absent)

        // Build options (only relevant when building from ONNX)
        bool   fp16            = true;
        bool   int8            = false;   // requires calibrator (not yet wired tho)
        size_t workspace_bytes = 1ULL << 30;  // 1 GiB
        int    device_id       = 0;

        // Runtime options
        int    max_batch_size  = 8;
        nvinfer1::ILogger::Severity log_level =
            nvinfer1::ILogger::Severity::kWARNING;
    };

    explicit trt_inference_engine(const config& cfg);
    ~trt_inference_engine();

    trt_inference_engine(const trt_inference_engine&)            = delete;
    trt_inference_engine& operator=(const trt_inference_engine&) = delete;
    trt_inference_engine(trt_inference_engine&&)                 = default;
    trt_inference_engine& operator=(trt_inference_engine&&)      = default;

    // Inference 

    // Raw: caller supplies / receives flat float32 host buffers (NCHW in, raw out).
    // Input buffer must have size N * C * H * W * sizeof(float).
    // Returns the raw output buffer (size depends on model).
    std::vector<float> run_raw(const std::vector<float>& input_blob,
                               int                       batch_size = 1);

    // High-level: YOLOv8-style detection.
    std::vector<detection>
    detect(const cv::Mat& image,
           float          conf_threshold = 0.5f,
           float          nms_threshold  = 0.45f);

    std::vector<std::vector<detection>>
    detect_batch(const std::vector<cv::Mat>& images,
                 float                       conf_threshold = 0.5f,
                 float                       nms_threshold  = 0.45f);

    void load_labels(const std::string& path);
    void set_labels(std::vector<std::string> labels);

    [[nodiscard]] cv::Size  net_input_size()  const noexcept { return net_size_; }
    [[nodiscard]] int       num_classes()     const noexcept { return num_classes_; }

private:
    // TRT objects
    trt_logger                                    logger_;
    std::unique_ptr<nvinfer1::IRuntime>           runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>        engine_;
    std::unique_ptr<nvinfer1::IExecutionContext>  context_;

    config           cfg_;
    cv::Size         net_size_   {640, 640};
    int              num_classes_{80};
    std::vector<std::string> labels_;

    // Device buffers (pinned to cfg_.device_id)
    void*  d_input_  = nullptr;
    void*  d_output_ = nullptr;
    size_t input_bytes_  = 0;
    size_t output_bytes_ = 0;

    // Build or deserialise the engine
    void build_or_load();
    void build_from_onnx();
    void load_from_file();
    void setup_io_buffers();
    void introspect();

    // Post-processing helpers (YOLOv8 decode + NMS)
    std::vector<detection>
    decode_yolov8(const float* out_data,
                  int          batch_idx,
                  int64_t      num_anchors,
                  int64_t      num_cols,
                  cv::Size     orig_size,
                  const struct letterbox_info& lb,
                  float        conf_thr,
                  float        nms_thr) const;
};


struct letterbox_info {
    float    scale;
    int      pad_left;
    int      pad_top;
    cv::Size orig_size;
};

} // namespace hpcv
