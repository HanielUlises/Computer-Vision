#include "torch_inference_engine.hpp"
#include "preprocessor.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace hpcv {

torch_inference_engine::torch_inference_engine(const config& cfg)
    : device_(cfg.device == config::device_type::cuda
              ? torch::Device(torch::kCUDA, cfg.device_id)
              : torch::Device(torch::kCPU))
    , cfg_(cfg)
{
    if (cfg_.num_threads > 0)
        torch::set_num_threads(cfg_.num_threads);

    load_module();
    introspect();
}

void torch_inference_engine::load_module()
{
    try {
        module_ = torch::jit::load(cfg_.model_path, device_);
    } catch (const c10::Error& e) {
        throw std::runtime_error(
            "[hpcv/torch] Failed to load model: " + cfg_.model_path +
            "\n  " + e.what());
    }

    module_.eval();

    if (cfg_.fp16 && cfg_.device == config::device_type::cuda) {
        module_.to(torch::kHalf);
        std::cout << "[hpcv/torch] FP16 enabled.\n";
    }

    std::cout << "[hpcv/torch] Model loaded: " << cfg_.model_path
              << "  device=" << device_name() << "\n";
}


void torch_inference_engine::introspect()
{

    try {
        torch::NoGradGuard no_grad;
        auto dummy = torch::zeros({1, 3, net_size_.height, net_size_.width},
                                  device_);
        if (cfg_.fp16 && cfg_.device == config::device_type::cuda)
            dummy = dummy.to(torch::kHalf);

        auto out = module_.forward({dummy});

        torch::Tensor t;
        if (out.isTensor()) {
            t = out.toTensor().to(torch::kFloat32).to(torch::kCPU);
        } else if (out.isTuple()) {
            t = out.toTuple()->elements()[0].toTensor()
                   .to(torch::kFloat32).to(torch::kCPU);
        }

        if (t.defined() && t.dim() == 3) {
            // Expected [N, 4+nc, anchors]
            int64_t ch = t.size(1);
            if (ch > 4) num_classes_ = (int)(ch - 4);
        }

        std::cout << "[hpcv/torch] Dummy forward OK. "
                  << "Classes: " << num_classes_ << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[hpcv/torch] Introspection warning: " << e.what() << "\n";
    }
}

std::string torch_inference_engine::device_name() const noexcept
{
    if (cfg_.device == config::device_type::cuda)
        return "cuda:" + std::to_string(cfg_.device_id);
    return "cpu";
}

void torch_inference_engine::load_labels(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("[hpcv/torch] Cannot open labels: " + path);
    labels_.clear();
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) labels_.push_back(line);
    std::cout << "[hpcv/torch] Loaded " << labels_.size() << " labels.\n";
}

void torch_inference_engine::set_labels(std::vector<std::string> l) {
    labels_ = std::move(l);
}


std::vector<torch::jit::IValue>
torch_inference_engine::run(const torch::Tensor& input)
{
    torch::NoGradGuard no_grad;

    auto t = input.to(device_);
    if (cfg_.fp16 && cfg_.device == config::device_type::cuda)
        t = t.to(torch::kHalf);

    auto out = module_.forward({t});

    std::vector<torch::jit::IValue> results;
    if (out.isTuple()) {
        for (auto& elem : out.toTuple()->elements())
            results.push_back(elem);
    } else {
        results.push_back(out);
    }
    return results;
}


std::vector<detection>
torch_inference_engine::detect(const cv::Mat& image,
                                float conf_thr, float nms_thr)
{
    auto batch = detect_batch({image}, conf_thr, nms_thr);
    return batch.empty() ? std::vector<detection>{} : std::move(batch[0]);
}

std::vector<std::vector<detection>>
torch_inference_engine::detect_batch(const std::vector<cv::Mat>& images,
                                      float conf_thr, float nms_thr)
{
    if (images.empty()) return {};
    int N = (int)images.size();

    std::vector<letterbox_info> lbs(N);
    std::vector<cv::Mat>        rgbs(N);
    for (int i = 0; i < N; ++i) {
        cv::cvtColor(images[i], rgbs[i], cv::COLOR_BGR2RGB);
        rgbs[i] = preprocessor::letterbox(rgbs[i], net_size_, &lbs[i]);
    }
    auto input_tensor = preprocessor::to_batch_tensor(rgbs, norm_zero_one());
    // input_tensor: [N, 3, H, W]  float32 CPU

    auto outputs = run(input_tensor);

    // YOLOv8: [N, 4+nc, anchors]
    if (outputs.empty() || !outputs[0].isTensor())
        throw std::runtime_error("[hpcv/torch] Unexpected model output type.");

    auto out_t = outputs[0].toTensor().to(torch::kFloat32).to(torch::kCPU);

    std::vector<std::vector<detection>> results(N);
    for (int b = 0; b < N; ++b) {
        results[b] = decode_yolov8(out_t, b, images[b].size(), lbs[b],
                                   conf_thr, nms_thr);
    }
    return results;
}

std::vector<detection>
torch_inference_engine::decode_yolov8(const torch::Tensor& output,
                                       int                  batch_idx,
                                       cv::Size             orig_size,
                                       const letterbox_info& lb,
                                       float                conf_thr,
                                       float                nms_thr) const
{
    // output: [N, 4+nc, anchors]
    auto batch_out = output[batch_idx];            // [4+nc, anchors]
    batch_out      = batch_out.contiguous();
    const float* data = batch_out.data_ptr<float>();

    int64_t cols    = batch_out.size(0);           // 4 + nc
    int64_t anchors = batch_out.size(1);
    int64_t nc      = cols - 4;

    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    std::vector<int>       class_ids;

    for (int64_t a = 0; a < anchors; ++a) {
        float cx = data[0 * anchors + a];
        float cy = data[1 * anchors + a];
        float bw = data[2 * anchors + a];
        float bh = data[3 * anchors + a];

        float best = -1e9f; int cls = 0;
        for (int64_t c = 0; c < nc; ++c) {
            float s = data[(4 + c) * anchors + a];
            if (s > best) { best = s; cls = (int)c; }
        }
        if (best < conf_thr) continue;

        boxes.emplace_back((int)(cx - bw*.5f), (int)(cy - bh*.5f),
                           (int)bw, (int)bh);
        scores.push_back(best);
        class_ids.push_back(cls);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thr, nms_thr, indices);

    std::vector<detection> dets;
    dets.reserve(indices.size());
    for (int idx : indices) {
        auto& b = boxes[idx];
        cv::Rect2f nb{(float)b.x, (float)b.y, (float)b.width, (float)b.height};
        auto un = preprocessor::unletterbox(nb, lb);

        detection d;
        d.class_id   = class_ids[idx];
        d.confidence = scores[idx];
        d.label      = (d.class_id < (int)labels_.size())
                       ? labels_[d.class_id]
                       : std::to_string(d.class_id);
        d.bbox = un;
        dets.push_back(d);
    }
    return dets;
}

} // namespace hpcv
