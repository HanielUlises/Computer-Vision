#include "blob_detector.h"

BlobDetector::BlobDetector(const cv::SimpleBlobDetector::Params& params)
    : detector_(cv::SimpleBlobDetector::create(params)) {}

std::vector<cv::KeyPoint> BlobDetector::detect(const cv::Mat& image) {
    std::vector<cv::KeyPoint> keypoints;
    detector_->detect(image, keypoints);
    return keypoints;
}

cv::Mat BlobDetector::draw_blobs(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints) {
    cv::Mat image_with_blobs;
    cv::drawKeypoints(image, keypoints, image_with_blobs, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return image_with_blobs;
}