#pragma once
// preprocessor.hpp

#include <opencv2/core.hpp>
#include <vector>
#include <cstdint>

namespace hpcv {

// Normalisation descriptors
struct norm_params {
    // :  out = (pixel / scale - mean) / std
    float scale = 255.0f;
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std [3] = {0.229f, 0.224f, 0.225f};
};

inline norm_params norm_image_net()  { return norm_params{255.f, {0.485f,0.456f,0.406f}, {0.229f,0.224f,0.225f}}; }
inline norm_params norm_zero_one()   { return norm_params{255.f, {0.f,0.f,0.f},           {1.f,1.f,1.f}};          }
inline norm_params norm_neg_one_one() { return norm_params{255.f, {0.5f,0.5f,0.5f},        {0.5f,0.5f,0.5f}};       }

// Padding meta (for letterbox)
struct letterbox_info {
    float  scale;
    int    pad_left;
    int    pad_top;
    cv::Size orig_size;
};

// preprocessor
// Stateless utility class. All methods are static.
class preprocessor {
public:
    // Letterbox resize (keeps aspect ratio, pads with grey 114)
    static cv::Mat letterbox(const cv::Mat& src,
                             cv::Size        dst_size,
                             letterbox_info*  info     = nullptr,
                             cv::Scalar      pad_color = {114, 114, 114});

    // Direct resize (fastest, distorts aspect ratio)
    static cv::Mat resize(const cv::Mat& src, cv::Size dst_size);

    // BGR->RGB + HWC->CHW + normalise -> flat float32 blob
    // Output vector has length C*H*W and is ready to be wrapped in an OrtValue.
    static std::vector<float> to_blob(const cv::Mat& rgb,
                                     const norm_params& norm = norm_zero_one());

    // Batch version: returns interleaved NCHW float32 blob
    static std::vector<float> to_batch_blob(const std::vector<cv::Mat>& rgbs,
                                          const norm_params& norm = norm_zero_one());

    // Inverse-letterbox: map a normalised bbox back to original image coords.
    static cv::Rect2f unletterbox(const cv::Rect2f& norm_box,
                                  const letterbox_info& info);
};

} // namespace hpcv