#include "patch_dataset.hpp"
#include <cstdlib>

namespace fs = std::filesystem;

PatchDataset::PatchDataset(const std::string& root) {
    int label = 0;
    for (const auto& dir : fs::directory_iterator(root)) {
        for (const auto& file : fs::directory_iterator(dir.path())) {
            samples_.emplace_back(file.path().string(), label);
        }
        ++label;
    }
}

static torch::Tensor load_patch(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    cv::resize(img, img, {32, 32});
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    return torch::from_blob(img.data, {1, 32, 32}).clone();
}

PatchSample PatchDataset::get_random() const {
    auto& s = samples_[rand() % samples_.size()];
    return { load_patch(s.first), s.second };
}

PatchSample PatchDataset::get_same(int label) const {
    while (true) {
        auto& s = samples_[rand() % samples_.size()];
        if (s.second == label)
            return { load_patch(s.first), s.second };
    }
}

PatchSample PatchDataset::get_diff(int label) const {
    while (true) {
        auto& s = samples_[rand() % samples_.size()];
        if (s.second != label)
            return { load_patch(s.first), s.second };
    }
}
