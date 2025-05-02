#pragma once
#include <expected>
#include <string>
#include <vector>
#include <span>
#include <Eigen/Dense>
#include <opencv2/core.hpp>

namespace camera_pose_estimation {
struct pose_result {
    Eigen::Matrix3d rotation;
    Eigen::Vector3d translation;
    double reprojection_error{0.0};
};

class HomographyPoseSolver {
public:
    HomographyPoseSolver(const Eigen::Matrix3d& camera_matrix,
                         const std::vector<double>& distortion_coeffs);

    std::expected<pose_result, std::string> estimate_pose(
        std::span<const cv::Point3d> world_points,
        std::span<const cv::Point2d> image_points);

    std::vector<cv::Point2d> project_points(
        std::span<const cv::Point3d> world_points, const pose_result& pose);

    // New: Homography-based calibration function
    static bool calibrate_homography(
        const std::vector<std::vector<cv::Point2f>>& image_points,
        const std::vector<std::vector<cv::Point3f>>& object_points,
        cv::Size image_size,
        Eigen::Matrix3d& camera_matrix,
        std::vector<double>& distortion_coeffs);

private:
    Eigen::Matrix3d camera_matrix;
    std::vector<double> distortion_coeffs;
};
} // namespace camera_pose_estimation