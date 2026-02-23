#pragma once

#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace hpcv {


struct detection {
    cv::Rect2f  bbox;        // normalised [0,1] (x, y, w, h)
    float       confidence;
    int         class_id;
    std::string label;
};


class torch_inference_engine {
public:
    enum class device_type { cpu, cuda };

    struct config {
        std::string  model_path;         // TorchScript
        device_type  device   = device_type::cuda;
        int          device_id = 0;
        bool         fp16      = false;  // half-precision on CUDA
        int          num_threads = 0;    // CPU thread pool; 0 = default
    };

    explicit torch_inference_engine(const config& cfg);
    ~torch_inference_engine() = default;

    torch_inference_engine(const torch_inference_engine&)            = delete;
    torch_inference_engine& operator=(const torch_inference_engine&) = delete;
    torch_inference_engine(torch_inference_engine&&)                 = default;
    torch_inference_engine& operator=(torch_inference_engine&&)      = default;

    // Raw inference 
    // Input:  NCHW float32 tensor (on CPU; engine moves it to the right device)
    // Output: vector of IValues returned by the TorchScript module
    std::vector<torch::jit::IValue> run(const torch::Tensor& input);

    // High-level detect
    std::vector<detection>
    detect(const cv::Mat& image,
           float          conf_threshold = 0.5f,
           float          nms_threshold  = 0.45f);

    std::vector<std::vector<detection>>
    detect_batch(const std::vector<cv::Mat>& images,
                 float                       conf_threshold = 0.5f,
                 float                       nms_threshold  = 0.45f);

    // Metadata 
    void load_labels(const std::string& path);
    void set_labels(std::vector<std::string> labels);

    [[nodiscard]] std::string  device_name()   const noexcept;
    [[nodiscard]] cv::Size     net_input_size() const noexcept { return net_size_; }

private:
    torch::jit::script::Module module_;
    torch::Device              device_;
    config                     cfg_;
    cv::Size                   net_size_   {640, 640};
    int                        num_classes_{80};
    std::vector<std::string>   labels_;

    void load_module();
    void introspect();

    // YOLOv8 decode + NMS
    std::vector<detection>
    decode_yolov8(const torch::Tensor& output,
                  int                  batch_idx,
                  cv::Size             orig_size,
                  const struct letterbox_info& lb,
                  float                conf_thr,
                  float                nms_thr) const;
};

struct letterbox_info {
    float    scale;
    int      pad_left;
    int      pad_top;
    cv::Size orig_size;
};

} // namespace hpcv
