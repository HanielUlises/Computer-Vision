#ifndef BLOB_DETECTOR_H
#define BLOB_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Dense>

class BlobDetector {
public:
    explicit BlobDetector(const cv::SimpleBlobDetector::Params& params);

    std::vector<cv::KeyPoint> detect(const cv::Mat& image);
    cv::Mat draw_blobs(const cv::Mat& image, const std::vector<cv::KeyPoint>& keypoints);

private:
    cv::Ptr<cv::SimpleBlobDetector> detector_;
};

#endif // BLOB_DETECTOR_H
