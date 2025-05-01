#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>
#include <expected>
#include <string>

namespace homography_stitching {
struct stitch_result {
    cv::Mat stitched_image;
    cv::Mat homography_matrix;
    int inlier_count;
};

class homography_stitcher {
public:
    homography_stitcher(const std::string& image_path_1, const std::string& image_path_2);
    std::expected<stitch_result, std::string> stitch_images();
    void save_stitched_image(const std::string& output_path);
private:
    std::vector<cv::KeyPoint> detect_keypoints(const cv::Mat& image);
    cv::Mat compute_descriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    std::vector<cv::DMatch> match_descriptors(const cv::Mat& descriptors_1, const cv::Mat& descriptors_2);
    std::optional<cv::Mat> estimate_homography(const std::vector<cv::KeyPoint>& keypoints_1,
                                              const std::vector<cv::KeyPoint>& keypoints_2,
                                              const std::vector<cv::DMatch>& matches);
    cv::Mat blend_images(const cv::Mat& image_1, const cv::Mat& warped_image_2);
    cv::Mat image_1;
    cv::Mat image_2;
    stitch_result result;
};
} // namespace homography_stitching