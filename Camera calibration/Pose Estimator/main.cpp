#include "camera_pose_estimator.hpp"
#include <iostream>
#include <opencv2/core.hpp>

int main() {
    try {
        // Sample camera matrix (fx, fy, cx, cy)
        Eigen::Matrix3d camera_matrix = Eigen::Matrix3d::Identity();
        camera_matrix(0, 0) = 800.0; // fx
        camera_matrix(1, 1) = 800.0; // fy
        camera_matrix(0, 2) = 320.0; // cx
        camera_matrix(1, 2) = 240.0; // cy

        // Sample distortion coefficients
        std::vector<double> distortion_coeffs = {0.1, -0.05, 0.001, 0.002, 0.0};

        // Initialize estimator
        camera_pose_estimation::camera_pose_estimator estimator(camera_matrix, distortion_coeffs);

        // Sample 3D world points (e.g., a square in 3D space)
        std::vector<cv::Point3d> world_points = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {1.0, 1.0, 0.0}
        };

        // Sample 2D image points (projected points, with some noise)
        std::vector<cv::Point2d> image_points = {
            {300.0, 200.0},
            {500.0, 200.0},
            {300.0, 400.0},
            {500.0, 400.0}
        };

        // Estimate pose
        auto result = estimator.estimate_pose(world_points, image_points);
        if (!result.has_value()) {
            std::cerr << "Pose estimation failed: " << result.error() << "\n";
            return 1;
        }

        // Output results
        const auto& pose = result.value();
        std::cout << "Rotation matrix:\n" << pose.rotation << "\n";
        std::cout << "Translation vector:\n" << pose.translation << "\n";
        std::cout << "Reprojection error: " << pose.reprojection_error << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}