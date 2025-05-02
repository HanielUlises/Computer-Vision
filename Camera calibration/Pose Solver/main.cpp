#include "homography_pose_solver.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    try {
        // Step 1: Simulate checkerboard calibration data
        cv::Size board_size(9, 6); // 9x6 inner corners
        float square_size = 0.025; // 25mm squares
        std::vector<std::vector<cv::Point3f>> object_points;
        std::vector<std::vector<cv::Point2f>> image_points;
        cv::Size image_size(640, 480);

        // 3D object points for a checkerboard
        std::vector<cv::Point3f> objp;
        for (int i = 0; i < board_size.height; ++i) {
            for (int j = 0; j < board_size.width; ++j) {
                objp.emplace_back(j * square_size, i * square_size, 0.0f);
            }
        }

        // Simulate 3 views of the checkerboard (in practice, detect corners from images)
        for (int i = 0; i < 3; ++i) {
            object_points.push_back(objp);
            std::vector<cv::Point2f> imgp;
            for (const auto& p : objp) {
                float x = p.x * 800 / square_size + 320 + (rand() % 10 - 5);
                float y = p.y * 800 / square_size + 240 + (rand() % 10 - 5);
                imgp.emplace_back(x, y);
            }
            image_points.push_back(imgp);
        }

        // Step 2: Calibrate using homography
        Eigen::Matrix3d camera_matrix;
        std::vector<double> distortion_coeffs;
        if (!HomographyPoseSolver::calibrate_homography(image_points, object_points, image_size,
                                                      camera_matrix, distortion_coeffs)) {
            std::cerr << "Calibration failed.\n";
            return 1;
        }

        // Step 3: Calibrated parameters for pose estimation
        HomographyPoseSolver solver(camera_matrix, distortion_coeffs);

        // Sample 3D world points (e.g., a square in 3D space)
        std::vector<cv::Point3d> world_points = {
            {0.0, 0.0, 0.0},
            {1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {1.0, 1.0, 0.0}
        };

        // Sample 2D image points (projected points, with some noise)
        std::vector<cv::Point2d> image_points_pose = {
            {300.0, 200.0},
            {500.0, 200.0},
            {300.0, 400.0},
            {500.0, 400.0}
        };

        // Pose estimation
        auto result = solver.estimate_pose(world_points, image_points_pose);
        if (!result.has_value()) {
            std::cerr << "Pose estimation failed: " << result.error() << "\n";
            return 1;
        }

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