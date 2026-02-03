#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>

class PatchDataset {
public:
    PatchDataset(const std::vector<std::string>& image_paths, int patch_size);

    torch::Tensor sampleBatch(int batch_size);

private:
    std::vector<cv::Mat> images_;
    int patch_size_;
};
