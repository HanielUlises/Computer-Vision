#pragma once
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <optional>
#include <span>
#include <expected>
#include <string>

namespace camera_pose_estimation {
struct pose_result {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    double reprojection_error;
};

class camera_pose_estimator {
public:
    camera_pose_estimator(const Eigen::Matrix3d& camera_matrix,
                         const std::vector<double>& distortion_coeffs);
    std::expected<pose_result, std::string> estimate_pose(
        std::span<const cv::Point3d> world_points,
        std::span<const cv::Point2d> image_points);
    std::vector<cv::Point2d> project_points(std::span<const cv::Point3d> world_points,
                                           const pose_result& pose);
    void save_annotated_image(const std::string& image_path,
                            std::span<const cv::Point2d> image_points,
                            std::span<const cv::Point2d> projected_points);
private:
    Eigen::Matrix3d camera_matrix;
    std::vector<double> distortion_coeffs;
    cv::Mat input_image;
};
} // namespace camera_pose_estimation