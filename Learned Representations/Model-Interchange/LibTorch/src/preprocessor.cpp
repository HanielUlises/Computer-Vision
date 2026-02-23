#include "preprocessor.hpp"

#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <algorithm>

namespace hpcv {

cv::Mat preprocessor::letterbox(const cv::Mat& src, cv::Size dst_size,
                                 letterbox_info* info, cv::Scalar pad_color)
{
    int src_w  = src.cols, src_h = src.rows;
    float scale   = std::min((float)dst_size.width  / src_w,
                             (float)dst_size.height / src_h);
    int scaled_w  = (int)std::round(src_w * scale);
    int scaled_h  = (int)std::round(src_h * scale);
    int pad_left  = (dst_size.width  - scaled_w) / 2;
    int pad_top   = (dst_size.height - scaled_h) / 2;

    cv::Mat scaled;
    cv::resize(src, scaled, {scaled_w, scaled_h}, 0, 0, cv::INTER_LINEAR);
    cv::Mat out(dst_size, src.type(), pad_color);
    scaled.copyTo(out(cv::Rect(pad_left, pad_top, scaled_w, scaled_h)));

    if (info) {
        info->scale     = scale;
        info->pad_left  = pad_left;
        info->pad_top   = pad_top;
        info->orig_size = {src_w, src_h};
    }
    return out;
}

cv::Mat preprocessor::resize(const cv::Mat& src, cv::Size dst_size)
{
    cv::Mat out;
    cv::resize(src, out, dst_size, 0, 0, cv::INTER_LINEAR);
    return out;
}

// Single image -> [1, C, H, W] float32 tensor 
torch::Tensor preprocessor::to_tensor(const cv::Mat& rgb, const norm_params& norm)
{
    if (rgb.type() != CV_8UC3)
        throw std::runtime_error("[hpcv] to_tensor expects CV_8UC3 (RGB)");

    int h = rgb.rows, w = rgb.cols;

    // Allocate [1, 3, H, W] tensor
    auto t = torch::empty({1, 3, h, w}, torch::kFloat32);
    float* data = t.data_ptr<float>();

    for (int y = 0; y < h; ++y) {
        const uint8_t* row = rgb.ptr<uint8_t>(y);
        for (int x = 0; x < w; ++x) {
            int px = y * w + x;
            // Channel plane offsets
            data[0 * h * w + px] = (row[x*3+0] / norm.scale - norm.mean[0]) / norm.std[0];
            data[1 * h * w + px] = (row[x*3+1] / norm.scale - norm.mean[1]) / norm.std[1];
            data[2 * h * w + px] = (row[x*3+2] / norm.scale - norm.mean[2]) / norm.std[2];
        }
    }
    return t;  // [1, 3, H, W]
}

// Batch → [N, C, H, W] tensor 
torch::Tensor preprocessor::to_batch_tensor(const std::vector<cv::Mat>& rgbs,
                                             const norm_params& norm)
{
    if (rgbs.empty()) return torch::empty({0});
    int n = (int)rgbs.size();
    std::vector<torch::Tensor> tensors;
    tensors.reserve(n);
    for (auto& img : rgbs)
        tensors.push_back(to_tensor(img, norm));   // [1, 3, H, W] each
    return torch::cat(tensors, 0);                  // [N, 3, H, W]
}

// Unletterbox 
cv::Rect2f preprocessor::unletterbox(const cv::Rect2f& nb, const letterbox_info& info)
{
    float x  = (nb.x      - info.pad_left) / info.scale / info.orig_size.width;
    float y  = (nb.y      - info.pad_top)  / info.scale / info.orig_size.height;
    float bw = nb.width   / info.scale / info.orig_size.width;
    float bh = nb.height  / info.scale / info.orig_size.height;
    return {std::clamp(x,  0.f, 1.f),
            std::clamp(y,  0.f, 1.f),
            std::clamp(bw, 0.f, 1.f - x),
            std::clamp(bh, 0.f, 1.f - y)};
}

} // namespace hpcv
