#pragma once

#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace hpcv {

enum class execution_provider {
    cpu,
    cuda,
    tensor_rt,  
    open_vino,
};

// Raw tensor descriptor
struct tensor_info {
    std::string               name;
    std::vector<int64_t>      shape;   // -1 means dynamic dim
    ONNXTensorElementDataType dtype;
};

// Inference result
struct detection {
    cv::Rect2f  bbox;       // normalised [0,1] coordinates
    float       confidence;
    int         class_id;
    std::string label;
};

class onnx_inference_engine {
public:
    struct config {
        std::string       model_path;
        execution_provider provider        = execution_provider::cuda;
        int               device_id        = 0;       // GPU index
        int               num_cpu_threads   = 0;       // 0 = ORT default
        bool              enable_profiling = false;
        size_t            cuda_mem_limit    = 0;       // bytes, 0 = no limit
        // Optimisation level: 0=disabled … 3=all
        int               graph_opt_level   = 3;
    };

    explicit onnx_inference_engine(const config& cfg);
    ~onnx_inference_engine() = default;

    onnx_inference_engine(const onnx_inference_engine&)            = delete;
    onnx_inference_engine& operator=(const onnx_inference_engine&) = delete;
    onnx_inference_engine(onnx_inference_engine&&)                 = default;
    onnx_inference_engine& operator=(onnx_inference_engine&&)      = default;

    [[nodiscard]] const std::vector<tensor_info>& input_infos()  const noexcept;
    [[nodiscard]] const std::vector<tensor_info>& output_infos() const noexcept;
    [[nodiscard]] std::string                    provider_name() const noexcept;

    // Raw tensor inference
    std::vector<Ort::Value> run(std::vector<Ort::Value>& inputs);

    std::vector<std::vector<detection>>
    detect_batch(const std::vector<cv::Mat>& images,
                float                       conf_threshold = 0.5f,
                float                       nms_threshold  = 0.45f);

    std::vector<detection>
    detect(const cv::Mat& image,
           float          conf_threshold = 0.5f,
           float          nms_threshold  = 0.45f);

    void load_labels(const std::string& labels_path);
    void set_labels(std::vector<std::string> labels);

private:
    Ort::Env                              env_;
    Ort::SessionOptions                   session_opts_;
    std::unique_ptr<Ort::Session>         session_;
    Ort::AllocatorWithDefaultOptions      allocator_;
    Ort::MemoryInfo                       mem_info_;

    config                                cfg_;
    std::vector<tensor_info>               input_infos_;
    std::vector<tensor_info>               output_infos_;
    std::vector<std::string>              labels_;

    std::vector<const char*>              input_names_;
    std::vector<const char*>              output_names_;

    void build_session();
    void introspect();
    void apply_provider();

    cv::Mat preprocess(const cv::Mat& src, cv::Size net_input_size) const;

    std::vector<detection>
    decode_detections(const float* data,
                     int64_t      rows,
                     int64_t      cols,
                     cv::Size     orig_size,
                     float        conf_threshold,
                     float        nms_threshold) const;
};

} // namespace hpcv