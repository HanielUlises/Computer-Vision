#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>

struct PatchSample {
    torch::Tensor patch;
    int label;
};

// Simple image-patch dataset.
// Directory structure:
// root/
//   class_0/*.png
//   class_1/*.png
class PatchDataset {
public:
    explicit PatchDataset(const std::string& root);

    PatchSample get_random() const;
    PatchSample get_same(int label) const;
    PatchSample get_diff(int label) const;

private:
    std::vector<std::pair<std::string, int>> samples_;
};
