#include "blob_detector.hpp"

BlobDetector::BlobDetector(const cv::SimpleBlobDetector::Params& params)
    : detector_(cv::SimpleBlobDetector::create(params)) {}

std::vector<cv::KeyPoint> BlobDetector::detect(const cv::Mat& image) {
    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(image, keypoints);
    return keypoints;
}

cv::Mat BlobDetector::draw_blobs(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat output;
    if (image.channels() == 1)
        cv::cvtColor(image, output, cv::COLOR_GRAY2BGR);
    else
        output = image.clone();

    for (const auto& kp : keypoints) {
        Eigen::Vector2f center(kp.pt.x, kp.pt.y);
        float radius = kp.size / 2.0f;

        cv::Point center_cv(static_cast<int>(center.x()), static_cast<int>(center.y()));
        int radius_cv = static_cast<int>(radius);

        cv::circle(output, center_cv, radius_cv, cv::Scalar(0, 0, 255), 2);
        cv::line(output,
                 cv::Point(center.x() - 3, center.y()),
                 cv::Point(center.x() + 3, center.y()),
                 cv::Scalar(255, 0, 0), 1);
        cv::line(output,
                 cv::Point(center.x(), center.y() - 3),
                 cv::Point(center.x(), center.y() + 3),
                 cv::Scalar(255, 0, 0), 1);
    }

    return output;
}
