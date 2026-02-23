#include "trt_inference_engine.hpp"
#include "preprocessor.hpp"

#include <NvOnnxParser.h>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace hpcv {


void trt_logger::log(Severity sev, const char* msg) noexcept
{
    if (sev > min_sev_) return;
    const char* tag = "[TRT]";
    switch (sev) {
    case Severity::kINTERNAL_ERROR: tag = "[TRT INTERNAL]"; break;
    case Severity::kERROR:          tag = "[TRT ERROR]";    break;
    case Severity::kWARNING:        tag = "[TRT WARN]";     break;
    case Severity::kINFO:           tag = "[TRT INFO]";     break;
    case Severity::kVERBOSE:        tag = "[TRT VERBOSE]";  break;
    }
    std::cerr << tag << " " << msg << "\n";
}


struct trt_deleter {
    template<typename T>
    void operator()(T* obj) const { if (obj) obj->destroy(); }
};


trt_inference_engine::trt_inference_engine(const config& cfg)
    : logger_(cfg.log_level), cfg_(cfg)
{
    cudaSetDevice(cfg_.device_id);
    build_or_load();
    setup_io_buffers();
    introspect();
}

trt_inference_engine::~trt_inference_engine()
{
    if (d_input_)  cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
}


void trt_inference_engine::build_or_load()
{
    bool have_engine = !cfg_.engine_path.empty() &&
                       std::ifstream(cfg_.engine_path).good();

    if (have_engine) {
        load_from_file();
    } else if (!cfg_.onnx_path.empty()) {
        build_from_onnx();
    } else {
        throw std::runtime_error(
            "[hpcv/trt] Provide either engine_path or onnx_path.");
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_)
        throw std::runtime_error("[hpcv/trt] Failed to create execution context.");
}

void trt_inference_engine::build_from_onnx()
{
    std::cout << "[hpcv/trt] Building engine from: " << cfg_.onnx_path << "\n";

    std::unique_ptr<nvinfer1::IBuilder,          trt_deleter> builder(
        nvinfer1::createInferBuilder(logger_));
    std::unique_ptr<nvinfer1::INetworkDefinition, trt_deleter> network(
        builder->createNetworkV2(
            1U << static_cast<uint32_t>(
                nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    std::unique_ptr<nvonnxparser::IParser, trt_deleter> parser(
        nvonnxparser::createParser(*network, logger_));

    if (!parser->parseFromFile(cfg_.onnx_path.c_str(),
                               static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    {
        throw std::runtime_error(
            "[hpcv/trt] ONNX parsing failed: " + cfg_.onnx_path);
    }

    // Config
    std::unique_ptr<nvinfer1::IBuilderConfig, trt_deleter> build_cfg(
        builder->createBuilderConfig());
    build_cfg->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                                  cfg_.workspace_bytes);

    if (cfg_.fp16 && builder->platformHasFastFp16()) {
        build_cfg->setFlag(nvinfer1::BuilderFlag::kFP16);
        std::cout << "[hpcv/trt] FP16 enabled.\n";
    }
    if (cfg_.int8 && builder->platformHasFastInt8()) {
        build_cfg->setFlag(nvinfer1::BuilderFlag::kINT8);
        std::cout << "[hpcv/trt] INT8 enabled (no calibrator supplied).\n";
    }

    // Serialise
    std::unique_ptr<nvinfer1::IHostMemory, trt_deleter> serialised(
        builder->buildSerializedNetwork(*network, *build_cfg));
    if (!serialised)
        throw std::runtime_error("[hpcv/trt] Engine serialisation failed.");

    if (!cfg_.engine_path.empty()) {
        std::ofstream f(cfg_.engine_path, std::ios::binary);
        f.write(static_cast<const char*>(serialised->data()),
                static_cast<std::streamsize>(serialised->size()));
        std::cout << "[hpcv/trt] Engine saved: " << cfg_.engine_path << "\n";
    }

    // Deserialise into a live engine
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(
        serialised->data(), serialised->size()));
    if (!engine_)
        throw std::runtime_error("[hpcv/trt] Failed to deserialise built engine.");
}

void trt_inference_engine::load_from_file()
{
    std::cout << "[hpcv/trt] Loading engine: " << cfg_.engine_path << "\n";
    std::ifstream f(cfg_.engine_path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error(
        "[hpcv/trt] Cannot open engine file: " + cfg_.engine_path);

    std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    if (!f.read(buf.data(), sz))
        throw std::runtime_error("[hpcv/trt] Failed to read engine file.");

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    engine_.reset(runtime_->deserializeCudaEngine(buf.data(), sz));
    if (!engine_)
        throw std::runtime_error("[hpcv/trt] Failed to deserialise engine.");
}


void trt_inference_engine::setup_io_buffers()
{
    // TensorRT 8+ uses tensor names for I/O
    // binding index 0 as input, 1 as output (standard for single-head YOLO).
    int n_bindings = engine_->getNbIOTensors();
    if (n_bindings < 2)
        throw std::runtime_error("[hpcv/trt] Engine must have >=2 I/O tensors.");

    auto get_vol = [&](const char* name) -> size_t {
        auto dims = engine_->getTensorShape(name);
        size_t vol = 1;
        for (int d = 0; d < dims.nbDims; ++d)
            vol *= (dims.d[d] > 0 ? dims.d[d] : cfg_.max_batch_size);
        return vol;
    };

    const char* in_name  = engine_->getIOTensorName(0);
    const char* out_name = engine_->getIOTensorName(1);

    input_bytes_  = get_vol(in_name)  * sizeof(float);
    output_bytes_ = get_vol(out_name) * sizeof(float);

    cudaMalloc(&d_input_,  input_bytes_);
    cudaMalloc(&d_output_, output_bytes_);

    context_->setTensorAddress(in_name,  d_input_);
    context_->setTensorAddress(out_name, d_output_);
}

void trt_inference_engine::introspect()
{
    // Network size from input binding shape [N, C, H, W]
    auto in_dims = engine_->getTensorShape(engine_->getIOTensorName(0));
    if (in_dims.nbDims == 4) {
        net_size_.height = (in_dims.d[2] > 0) ? in_dims.d[2] : 640;
        net_size_.width  = (in_dims.d[3] > 0) ? in_dims.d[3] : 640;
    }

    // Number of classes from output binding shape [N, 4+nc, anchors]
    auto out_dims = engine_->getTensorShape(engine_->getIOTensorName(1));
    if (out_dims.nbDims == 3 && out_dims.d[1] > 4) {
        num_classes_ = (int)(out_dims.d[1] - 4);
    }

    std::cout << "[hpcv/trt] Engine ready. "
              << "Input: " << net_size_.width << "x" << net_size_.height
              << "  Classes: " << num_classes_ << "\n";
}


std::vector<float> trt_inference_engine::run_raw(const std::vector<float>& blob,
                                                  int batch_size)
{
    if (blob.size() * sizeof(float) > input_bytes_)
        throw std::runtime_error("[hpcv/trt] Input blob exceeds allocated buffer.");

    // Host -> device
    cudaMemcpy(d_input_, blob.data(),
               blob.size() * sizeof(float), cudaMemcpyHostToDevice);

    auto in_name = engine_->getIOTensorName(0);
    auto dims    = context_->getTensorShape(in_name);
    if (dims.d[0] < 0) {
        dims.d[0] = batch_size;
        context_->setInputShape(in_name, dims);
    }

    if (!context_->executeV2(nullptr))
        throw std::runtime_error("[hpcv/trt] executeV2 failed.");

    // Device -> host
    size_t out_elems = output_bytes_ / sizeof(float);
    std::vector<float> result(out_elems);
    cudaMemcpy(result.data(), d_output_, output_bytes_, cudaMemcpyDeviceToHost);
    return result;
}


std::vector<detection>
trt_inference_engine::detect(const cv::Mat& image,
                              float conf_thr, float nms_thr)
{
    auto batch = detect_batch({image}, conf_thr, nms_thr);
    return batch.empty() ? std::vector<detection>{} : std::move(batch[0]);
}

std::vector<std::vector<detection>>
trt_inference_engine::detect_batch(const std::vector<cv::Mat>& images,
                                    float conf_thr, float nms_thr)
{
    if (images.empty()) return {};
    int N = (int)images.size();
    if (N > cfg_.max_batch_size)
        throw std::runtime_error("[hpcv/trt] Batch exceeds max_batch_size.");

    std::vector<letterbox_info> lbs(N);
    std::vector<cv::Mat>        rgbs(N);
    for (int i = 0; i < N; ++i) {
        cv::cvtColor(images[i], rgbs[i], cv::COLOR_BGR2RGB);
        rgbs[i] = preprocessor::letterbox(rgbs[i], net_size_, &lbs[i]);
    }
    auto blob = preprocessor::to_batch_blob(rgbs, norm_zero_one());

    // Infer
    auto out = run_raw(blob, N);

    // Decode
    // YOLOv8 output layout: [N, 4+nc, anchors]
    auto out_dims   = engine_->getTensorShape(engine_->getIOTensorName(1));
    int64_t cols    = out_dims.d[1];   // 4 + nc
    int64_t anchors = out_dims.d[2];

    std::vector<std::vector<detection>> results(N);
    for (int b = 0; b < N; ++b) {
        const float* ptr = out.data() + b * cols * anchors;
        results[b] = decode_yolov8(ptr, b, anchors, cols,
                                   images[b].size(), lbs[b],
                                   conf_thr, nms_thr);
    }
    return results;
}

// YOLOv8 decode + NMS

std::vector<detection>
trt_inference_engine::decode_yolov8(const float* data,
                                     int          /*batch_idx*/,
                                     int64_t      anchors,
                                     int64_t      cols,
                                     cv::Size     orig_size,
                                     const letterbox_info& lb,
                                     float        conf_thr,
                                     float        nms_thr) const
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

        float best = -1e9f; int cls = 0;
        for (int64_t c = 0; c < nc; ++c) {
            float s = data[(4 + c) * anchors + a];
            if (s > best) { best = s; cls = (int)c; }
        }
        if (best < conf_thr) continue;

        boxes.emplace_back((int)(cx - bw * .5f), (int)(cy - bh * .5f),
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
        // Unletterbox
        cv::Rect2f nb{(float)b.x, (float)b.y, (float)b.width, (float)b.height};
        auto un = preprocessor::unletterbox(nb, lb);

        detection d;
        d.class_id   = class_ids[idx];
        d.confidence = scores[idx];
        d.label      = (d.class_id < (int)labels_.size())
                       ? labels_[d.class_id]
                       : std::to_string(d.class_id);
        d.bbox       = un;
        dets.push_back(d);
    }
    return dets;
}

void trt_inference_engine::load_labels(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("[hpcv/trt] Cannot open labels: " + path);
    labels_.clear();
    std::string line;
    while (std::getline(f, line))
        if (!line.empty()) labels_.push_back(line);
    std::cout << "[hpcv/trt] Loaded " << labels_.size() << " labels.\n";
}

void trt_inference_engine::set_labels(std::vector<std::string> l) {
    labels_ = std::move(l);
}

} // namespace hpcv
