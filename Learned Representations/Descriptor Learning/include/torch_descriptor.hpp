#pragma once
#include <opencv2/core.hpp>
#include "patch_net.hpp"
#include "embedding.hpp"

class TorchPatchDescriptor {
public:
    explicit TorchPatchDescriptor(const std::string& model_path);

    Embedding compute(
        const cv::Mat& image,
        const cv::KeyPoint& kp
    );

private:
    PatchNet net_;
};
