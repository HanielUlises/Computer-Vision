#include "camera_pose_estimator.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <ranges>
#include <numeric>

namespace camera_pose_estimation {
camera_pose_estimator::camera_pose_estimator(const Eigen::Matrix3d& camera_matrix,
                                           const std::vector<double>& distortion_coeffs)
    : camera_matrix(camera_matrix), distortion_coeffs(distortion_coeffs) {}

std::expected<pose_result, std::string> camera_pose_estimator::estimate_pose(
    std::span<const cv::Point3d> world_points,
    std::span<const cv::Point2d> image_points) {
    if (world_points.size() != image_points.size() || world_points.size() < 4) {
        return std::unexpected("invalid_point_count");
    }
    cv::Mat rvec, tvec;
    cv::Mat dist_coeffs = cv::Mat(distortion_coeffs, true);
    cv::Mat camera_matrix_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            camera_matrix_cv.at<double>(i, j) = camera_matrix(i, j);
        }
    }
    bool success = cv::solvePnP(world_points, image_points, camera_matrix_cv,
                               dist_coeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    if (!success) {
        return std::unexpected("pnp_solver_failed");
    }
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    pose_result result;
    result.rotation = Eigen::Matrix3d::Zero();
    result.translation = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; ++i) {
        result.translation(i) = tvec.at<double>(i);
        for (int j = 0; j < 3; ++j) {
            result.rotation(i, j) = rotation_matrix.at<double>(i, j);
        }
    }
    auto projected_points = project_points(world_points, result);
    result.reprojection_error = std::transform_reduce(
        image_points.begin(), image_points.end(), projected_points.begin(), 0.0,
        std::plus{}, [](const auto& p1, const auto& p2) {
            return std::hypot(p1.x - p2.x, p1.y - p2.y);
        }) / image_points.size();
    return result;
}

std::vector<cv::Point2d> camera_pose_estimator::project_points(
    std::span<const cv::Point3d> world_points, const pose_result& pose) {
    cv::Mat rvec, tvec(3, 1, CV_64F), rotation_matrix(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        tvec.at<double>(i) = pose.translation(i);
        for (int j = 0; j < 3; ++j) {
            rotation_matrix.at<double>(i, j) = pose.rotation(i, j);
        }
    }
    cv::Rodrigues(rotation_matrix, rvec);
    cv::Mat camera_matrix_cv(3, 3, CV_64F);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            camera_matrix_cv.at<double>(i, j) = camera_matrix(i, j);
        }
    }
    cv::Mat dist_coeffs = cv::Mat(distortion_coeffs, true);
    std::vector<cv::Point2d> projected_points;
    cv::projectPoints(world_points, rvec, tvec, camera_matrix_cv,
                     dist Reprojection error: {result.reprojection_error:.4f}\n");
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}