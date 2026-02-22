#include "onnx_inference_engine.hpp"
#include "preprocessor.hpp"

#include <onnxruntime/core/providers/cuda/cuda_provider_factory.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <sstream>

namespace hpcv {

onnx_inference_engine::onnx_inference_engine(const config& cfg)
    : env_(ORT_LOGGING_LEVEL_WARNING, "hpcv")
    , mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    , cfg_(cfg)
{
    build_session();
    introspect();
}

// Session construction
void onnx_inference_engine::build_session()
{
    // Graph optimisation
    const OrtGraphOptimizationLevel opt_levels[] = {
        ORT_DISABLE_ALL, ORT_ENABLE_BASIC, ORT_ENABLE_EXTENDED, ORT_ENABLE_ALL
    };
    int lvl = std::clamp(cfg_.graph_opt_level, 0, 3);
    session_opts_.SetGraphOptimizationLevel(opt_levels[lvl]);

    // CPU thread pool
    if (cfg_.num_cpu_threads > 0)
        session_opts_.SetIntraOpNumThreads(cfg_.num_cpu_threads);

    // Profiling
    if (cfg_.enable_profiling)
        session_opts_.EnableProfiling("hpcv_ort_profile_");

    apply_provider();

    session_ = std::make_unique<Ort::Session>(
        env_, cfg_.model_path.c_str(), session_opts_);
}

void onnx_inference_engine::apply_provider()
{
    switch (cfg_.provider) {
    case execution_provider::cuda: {
        OrtCUDAProviderOptions cuda_opts{};
        cuda_opts.device_id                 = cfg_.device_id;
        cuda_opts.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
        cuda_opts.gpu_mem_limit             = cfg_.cuda_mem_limit;
        cuda_opts.arena_extend_strategy     = 1;  // kSameAsRequested
        session_opts_.AppendExecutionProvider_CUDA(cuda_opts);
        std::cout << "[hpcv] ONNX backend: CUDA (device " << cfg_.device_id << ")\n";
        break;
    }
    case execution_provider::tensor_rt: {
        // TRT EP, falls back to CUDA automatically on failure
        OrtTensorRTProviderOptions trt_opts{};
        trt_opts.device_id = cfg_.device_id;
        trt_opts.trt_fp16_enable = 1;
        try {
            session_opts_.AppendExecutionProvider_TensorRT(trt_opts);
            std::cout << "[hpcv] ONNX backend: TensorRT (device " << cfg_.device_id << ")\n";
        } catch (const Ort::Exception& e) {
            std::cerr << "[hpcv] TensorRT unavailable (" << e.what()
                      << "), falling back to CUDA.\n";
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = cfg_.device_id;
            session_opts_.AppendExecutionProvider_CUDA(cuda_opts);
        }
        break;
    }
    case execution_provider::open_vino: {
        try {
            session_opts_.AppendExecutionProvider("OpenVINO", {}, {}, 0);
            std::cout << "[hpcv] ONNX backend: OpenVINO\n";
        } catch (...) {
            std::cerr << "[hpcv] OpenVINO unavailable, falling back to CPU.\n";
        }
        break;
    }
    case execution_provider::cpu:
    default:
        std::cout << "[hpcv] ONNX backend: CPU\n";
        break;
    }
}

// Model introspection
static tensor_info parse_tensor_info(Ort::Session& session,
                                  Ort::AllocatorWithDefaultOptions& alloc,
                                  bool input, size_t idx)
{
    tensor_info info;

    auto name_ptr = input ? session.GetInputNameAllocated(idx, alloc)
                         : session.GetOutputNameAllocated(idx, alloc);
    info.name = name_ptr.get();

    auto type_info = input ? session.GetInputTypeInfo(idx)
                          : session.GetOutputTypeInfo(idx);
    auto tensor_type_info = type_info.GetTensorTypeAndShapeInfo();

    info.shape = tensor_type_info.GetShape();
    info.dtype = tensor_type_info.GetElementType();
    return info;
}

void onnx_inference_engine::introspect()
{
    size_t n_in  = session_->GetInputCount();
    size_t n_out = session_->GetOutputCount();

    input_infos_.resize(n_in);
    output_infos_.resize(n_out);
    input_names_.resize(n_in);
    output_names_.resize(n_out);

    for (size_t i = 0; i < n_in; ++i) {
        input_infos_[i] = parse_tensor_info(*session_, allocator_, true, i);
        input_names_[i] = input_infos_[i].name.c_str();
    }
    for (size_t i = 0; i < n_out; ++i) {
        output_infos_[i] = parse_tensor_info(*session_, allocator_, false, i);
        output_names_[i] = output_infos_[i].name.c_str();
    }

    std::cout << "[hpcv] Model loaded: " << cfg_.model_path << "\n";
    for (auto& t : input_infos_) {
        std::cout << "  in  '" << t.name << "' shape=[";
        for (size_t i = 0; i < t.shape.size(); ++i)
            std::cout << (i ? "," : "") << t.shape[i];
        std::cout << "]\n";
    }
    for (auto& t : output_infos_) {
        std::cout << "  out '" << t.name << "' shape=[";
        for (size_t i = 0; i < t.shape.size(); ++i)
            std::cout << (i ? "," : "") << t.shape[i];
        std::cout << "]\n";
    }
}

// Accessors
const std::vector<tensor_info>& onnx_inference_engine::input_infos()  const noexcept { return input_infos_;  }
const std::vector<tensor_info>& onnx_inference_engine::output_infos() const noexcept { return output_infos_; }

std::string onnx_inference_engine::provider_name() const noexcept {
    switch (cfg_.provider) {
    case execution_provider::cuda:      return "CUDA";
    case execution_provider::tensor_rt:  return "TensorRT";
    case execution_provider::open_vino:  return "OpenVINO";
    default:                           return "CPU";
    }
}

// Raw run
std::vector<Ort::Value> onnx_inference_engine::run(std::vector<Ort::Value>& inputs)
{
    return session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        inputs.data(),
        inputs.size(),
        output_names_.data(),
        output_names_.size());
}

// High-level detect (single image)
std::vector<detection>
onnx_inference_engine::detect(const cv::Mat& image,
                             float          conf_threshold,
                             float          nms_threshold)
{
    auto results = detect_batch({image}, conf_threshold, nms_threshold);
    return results.empty() ? std::vector<detection>{} : std::move(results[0]);
}

// High-level detectBatch
std::vector<std::vector<detection>>
onnx_inference_engine::detect_batch(const std::vector<cv::Mat>& images,
                                  float conf_threshold,
                                  float nms_threshold)
{
    if (images.empty()) return {};

    // Determine network input size from model metadata
    // Shape is expected [N, C, H, W]; dynamic dims = -1
    const auto& in_shape = input_infos_[0].shape;
    int net_h = (in_shape.size() >= 4 && in_shape[2] > 0) ? (int)in_shape[2] : 640;
    int net_w = (in_shape.size() >= 4 && in_shape[3] > 0) ? (int)in_shape[3] : 640;
    cv::Size net_size{net_w, net_h};

    int batch_size = (int)images.size();

    // Preprocess
    std::vector<letterbox_info> lb_infos(batch_size);
    std::vector<cv::Mat> resized(batch_size);

    for (int i = 0; i < batch_size; ++i) {
        cv::Mat rgb;
        cv::cvtColor(images[i], rgb, cv::COLOR_BGR2RGB);
        resized[i] = preprocessor::letterbox(rgb, net_size, &lb_infos[i]);
    }

    std::vector<float> blob = preprocessor::to_batch_blob(resized, norm_zero_one());

    // Build input OrtValue
    std::vector<int64_t> input_shape = {batch_size, 3, net_h, net_w};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        mem_info_,
        blob.data(),
        blob.size(),
        input_shape.data(),
        input_shape.size());

    std::vector<Ort::Value> input_vec;
    input_vec.push_back(std::move(input_tensor));

    // Run inference
    auto outputs = run(input_vec);

    // Decode outputs
    // YOLOv8 output: [batch, num_classes+4, num_anchors]
    //   -> transposed to [batch, num_anchors, num_classes+4] for convenience
    auto& out_tensor    = outputs[0];
    auto  out_shape     = out_tensor.GetTensorTypeAndShapeInfo().GetShape();
    const float* data  = out_tensor.GetTensorData<float>();

    // out_shape = [N, 4+nc, anchors]
    int64_t nc       = out_shape[1] - 4;
    int64_t anchors  = out_shape[2];

    std::vector<std::vector<detection>> results(batch_size);
    for (int b = 0; b < batch_size; ++b) {
        const float* batch_data = data + b * (out_shape[1] * anchors);
        results[b] = decode_detections(
            batch_data, anchors, out_shape[1],
            images[b].size(), conf_threshold, nms_threshold);

        // Unletterbox back to original coords
        for (auto& det : results[b])
            det.bbox = preprocessor::unletterbox(det.bbox, lb_infos[b]);
    }

    return results;
}

// YOLOv8 decode + NMS
std::vector<detection>
onnx_inference_engine::decode_detections(const float* data,
                                       int64_t      anchors,
                                       int64_t      cols,
                                       cv::Size     orig_size,
                                       float        conf_threshold,
                                       float        nms_threshold) const
{
    int64_t nc = cols - 4;

    std::vector<cv::Rect>  boxes;
    std::vector<float>     scores;
    std::vector<int>       class_ids;

    for (int64_t a = 0; a < anchors; ++a) {
        float cx = data[0 * anchors + a];
        float cy = data[1 * anchors + a];
        float bw = data[2 * anchors + a];
        float bh = data[3 * anchors + a];

        // Best class score
        float max_score = -1e9f;
        int   best_cls  = 0;
        for (int64_t c = 0; c < nc; ++c) {
            float s = data[(4 + c) * anchors + a];
            if (s > max_score) { max_score = s; best_cls = (int)c; }
        }

        if (max_score < conf_threshold) continue;

        // cx,cy,w,h -> x,y,w,h in pixel coords
        int x = (int)((cx - bw * 0.5f));
        int y = (int)((cy - bh * 0.5f));
        int w = (int)(bw);
        int h = (int)(bh);

        boxes.emplace_back(x, y, w, h);
        scores.push_back(max_score);
        class_ids.push_back(best_cls);
    }

    // OpenCV NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, nms_threshold, indices);

    std::vector<detection> detections;
    detections.reserve(indices.size());

    for (int idx : indices) {
        detection det;
        auto&     b    = boxes[idx];
        det.confidence = scores[idx];
        det.class_id    = class_ids[idx];
        det.label      = (det.class_id < (int)labels_.size())
                         ? labels_[det.class_id]
                         : std::to_string(det.class_id);

        // Normalise to [0,1]
        det.bbox = {
            b.x / (float)orig_size.width,
            b.y / (float)orig_size.height,
            b.width  / (float)orig_size.width,
            b.height / (float)orig_size.height
        };
        detections.push_back(det);
    }

    return detections;
}

// Label loading
void onnx_inference_engine::load_labels(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("[hpcv] Cannot open labels: " + path);
    labels_.clear();
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) labels_.push_back(line);
    std::cout << "[hpcv] Loaded " << labels_.size() << " labels.\n";
}

void onnx_inference_engine::set_labels(std::vector<std::string> labels) {
    labels_ = std::move(labels);
}

} // namespace hpcv