#ifndef MEAN_SHIFT_HPP
#define MEAN_SHIFT_HPP

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

class MeanShift {
public:
    MeanShift(double bandwidth, int max_iter = 300, double tol = 1e-3);
    void fit(const std::vector<cv::Mat>& descriptors);
    Eigen::MatrixXd get_modes() const;

private:
    double bandwidth_;      // kernel radius h
    int max_iter_;          // maximum iterations
    double tol_;            // convergence threshold ε
    Eigen::MatrixXd modes_; // each row is a mode

    std::vector<Eigen::VectorXd> convert_to_vectors(const std::vector<cv::Mat>& descriptors);
    Eigen::VectorXd shift_point(const Eigen::VectorXd& point, const std::vector<Eigen::VectorXd>& data_points);
    double gaussian_kernel(double distance_sq) const;
};

#endif // MEAN_SHIFT_HPP
