#include "mean_shift.hpp"
#include <cmath>
#include <limits>
#include <unordered_set>

MeanShift::MeanShift(double bandwidth, int max_iter, double tol)
    : bandwidth_(bandwidth), max_iter_(max_iter), tol_(tol) {}

// Convert set of OpenCV descriptor matrices to Eigen vectors (still -> hierr)
std::vector<Eigen::VectorXd> MeanShift::convert_to_vectors(const std::vector<cv::Mat>& descriptors) {
    std::vector<Eigen::VectorXd> data_points;
    for (const auto& desc : descriptors) {
        for (int i = 0; i < desc.rows; ++i) {
            Eigen::VectorXd point(desc.cols);
            for (int j = 0; j < desc.cols; ++j) {
                point(j) = static_cast<double>(desc.at<float>(i, j));
            }
            data_points.push_back(point);
        }
    }
    return data_points;
}

// Applies mean shift algorithm to the data
void MeanShift::fit(const std::vector<cv::Mat>& descriptors) {
    auto data_points = convert_to_vectors(descriptors);
    const int n = static_cast<int>(data_points.size());
    const int d = data_points[0].size();

    std::vector<Eigen::VectorXd> shifted_points = data_points;

    for (int i = 0; i < n; ++i) {
        int iter = 0;
        Eigen::VectorXd current = shifted_points[i];

        while (iter < max_iter_) {
            Eigen::VectorXd new_point = shift_point(current, data_points);
            if ((new_point - current).norm() < tol_) break;
            current = new_point;
            ++iter;
        }

        shifted_points[i] = current;
    }

    // Mode consolidation: cluster points with pairwise distance < bandwidth / 2
    std::vector<Eigen::VectorXd> unique_modes;
    for (const auto& p : shifted_points) {
        bool is_new = true;
        for (const auto& m : unique_modes) {
            if ((p - m).norm() < bandwidth_ / 2) {
                is_new = false;
                break;
            }
        }
        if (is_new) unique_modes.push_back(p);
    }

    modes_ = Eigen::MatrixXd(unique_modes.size(), d);
    for (size_t i = 0; i < unique_modes.size(); ++i) {
        modes_.row(i) = unique_modes[i].transpose();
    }
}

// I know non of this will render in github .l.
// Compute shift vector: m(x) ← (Σ_{i} K(‖x - xᵢ‖²)·xᵢ) / (Σ_{i} K(‖x - xᵢ‖²))
Eigen::VectorXd MeanShift::shift_point(const Eigen::VectorXd& point, const std::vector<Eigen::VectorXd>& data_points) {
    Eigen::VectorXd numerator = Eigen::VectorXd::Zero(point.size());
    double denominator = 0.0;

    for (const auto& xi : data_points) {
        double dist_sq = (point - xi).squaredNorm();
        double weight = gaussian_kernel(dist_sq);
        numerator += weight * xi;
        denominator += weight;
    }

    return denominator > 0 ? numerator / denominator : point;
}

// Gaussian kernel: K(u) = exp(-u / (2h²))
double MeanShift::gaussian_kernel(double distance_sq) const {
    double h2 = bandwidth_ * bandwidth_;
    return std::exp(-distance_sq / (2.0 * h2));
}

// Final mode matrix
Eigen::MatrixXd MeanShift::get_modes() const {
    return modes_;
}
