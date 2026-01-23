#include "torch_descriptor.hpp"
#include <opencv2/imgproc.hpp>

TorchPatchDescriptor::TorchPatchDescriptor(
    const std::string& model_path
) {
    torch::load(net_, model_path);
    net_->eval();
}

Embedding TorchPatchDescriptor::compute(
    const cv::Mat& image,
    const cv::KeyPoint& kp
) const {
    cv::Rect roi(
        kp.pt.x - 16,
        kp.pt.y - 16,
        32, 32
    );
    roi &= cv::Rect(0, 0, image.cols, image.rows);

    cv::Mat patch = image(roi);
    cv::cvtColor(patch, patch, cv::COLOR_BGR2GRAY);
    cv::resize(patch, patch, {32, 32});
    patch.convertTo(patch, CV_32F, 1.0 / 255.0);

    auto input = torch::from_blob(
        patch.data, {1, 1, 32, 32}
    ).clone();

    torch::NoGradGuard guard;
    auto out = net_->forward(input).squeeze();

    Embedding e(128);
    std::memcpy(
        e.data.data(),
        out.data_ptr<float>(),
        128 * sizeof(float)
    );
    return e;
}
