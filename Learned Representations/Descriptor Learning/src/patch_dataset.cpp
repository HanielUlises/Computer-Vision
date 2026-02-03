#include "patch_dataset.hpp"
#include <random>

PatchDataset::PatchDataset(
    const std::vector<std::string>& image_paths,
    int patch_size
) : patch_size_(patch_size) {
    for (const auto& p : image_paths) {
        images_.push_back(cv::imread(p, cv::IMREAD_GRAYSCALE));
    }
}

torch::Tensor PatchDataset::sampleBatch(int batch_size) {
    std::vector<torch::Tensor> batch;
    std::mt19937 rng(std::random_device{}());

    for (int i = 0; i < batch_size; ++i) {
        const auto& img = images_[rng() % images_.size()];
        int x = rng() % (img.cols - patch_size_);
        int y = rng() % (img.rows - patch_size_);

        cv::Mat patch = img(cv::Rect(x, y, patch_size_, patch_size_));
        auto t = torch::from_blob(
            patch.data,
            {1, patch_size_, patch_size_},
            torch::kUInt8
        ).to(torch::kFloat32) / 255.0;

        batch.push_back(t);
    }

    return torch::stack(batch);
}
