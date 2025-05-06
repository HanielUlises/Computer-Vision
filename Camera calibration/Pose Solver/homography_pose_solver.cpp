#include "homography_pose_solver.hpp"
#include <opencv2/calib3d.hpp>
#include <ranges>
#include <numeric>

namespace camera_pose_estimation {
HomographyPoseSolver::HomographyPoseSolver(const Eigen::Matrix3d& camera_matrix,
                                         const std::vector<double>& distortion_coeffs)
    : camera_matrix(camera_matrix), distortion_coeffs(distortion_coeffs) {}

std::expected<pose_result, std::string> HomographyPoseSolver::estimate_pose(
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

std::vector<cv::Point2d> HomographyPoseSolver::project_points(
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
    cv::projectPoints(world_points, rvec, tvec, camera_matrix_cv, dist_coeffs, projected_points);
    return projected_points;
}

bool HomographyPoseSolver::calibrate_homography(
    const std::vector<std::vector<cv::Point2f>>& image_points,
    const std::vector<std::vector<cv::Point3f>>& object_points,
    cv::Size image_size,
    Eigen::Matrix3d& camera_matrix,
    std::vector<double>& distortion_coeffs) {
    if (image_points.size() < 3 || image_points.size() != object_points.size()) {
        std::cerr << "Need at least 3 views for calibration.\n";
        return false;
    }

    // Homographies
    std::vector<cv::Mat> homographies;
    for (size_t i = 0; i < image_points.size(); ++i) {
        cv::Mat H = cv::findHomography(object_points[i], image_points[i], cv::RANSAC);
        if (!H.empty()) {
            homographies.push_back(H);
        }
    }

    if (homographies.size() < 3) {
        std::cerr << "Insufficient valid homographies.\n";
        return false;
    }

    cv::Mat K, dist;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(object_points, image_points, image_size, K, dist, rvecs, tvecs,
                                     cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_ZERO_TANGENT_DIST);

    camera_matrix = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            camera_matrix(i, j) = K.at<double>(i, j);
        }
    }
    distortion_coeffs.assign(dist.begin<double>(), dist.end<double>());

    std::cout << "Calibration RMS error: " << rms << "\n";
    std::cout << "Camera matrix:\n" << camera_matrix << "\n";
    std::cout << "Distortion coefficients: ";
    for (const auto& d : distortion_coeffs) std::cout << d << " ";
    std::cout << "\n";

    return rms < 1.0; // This threshold is arbitrary for "acceptable" calibration
}
} // namespace camera_pose_estimation