#include "kmeans.hpp"
#include <random>
#include <limits>
#include <iostream>

KMeans::KMeans(int k, int max_iter, double tol)
    : k_(k), max_iter_(max_iter), tol_(tol) {}

// Convert set of OpenCV descriptor matrices to Eigen vectors
std::vector<Eigen::VectorXd> KMeans::convert_to_vectors(const std::vector<cv::Mat>& descriptors) {
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

// Lloyd's algorithm for k-means clustering
void KMeans::fit(const std::vector<cv::Mat>& descriptors) {
    auto data_points = convert_to_vectors(descriptors);
    int d = data_points[0].size(); // dimension of input vectors

    // Initialize centroids by sampling k points uniformly at random
    centroids_ = Eigen::MatrixXd(k_, d);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, static_cast<int>(data_points.size() - 1));
    for (int i = 0; i < k_; ++i) {
        centroids_.row(i) = data_points[distrib(gen)].transpose();
    }

    std::vector<int> labels(data_points.size(), -1);
    bool converged = false;
    int iter = 0;

    while (!converged && iter < max_iter_) {
        std::vector<int> new_labels(data_points.size());
        converged = true;

        // Assignment step: for each xᵢ, find argmin_j ||xᵢ - μⱼ||
        for (size_t i = 0; i < data_points.size(); ++i) {
            double min_dist = std::numeric_limits<double>::max();
            int best_j = -1;
            for (int j = 0; j < k_; ++j) {
                double dist = compute_distance(data_points[i], centroids_.row(j).transpose());
                if (dist < min_dist) {
                    min_dist = dist;
                    best_j = j;
                }
            }
            new_labels[i] = best_j;
            if (new_labels[i] != labels[i]) {
                converged = false;
            }
        }

        // Update step: μⱼ ← (1 / |Cⱼ|) Σ_{xᵢ ∈ Cⱼ} xᵢ
        update_centroids(data_points, new_labels);
        labels = new_labels;
        ++iter;
    }
}

// Predict the nearest cluster index for each point
std::vector<int> KMeans::predict(const std::vector<cv::Mat>& descriptors) {
    auto data_points = convert_to_vectors(descriptors);
    std::vector<int> predictions;

    for (const auto& point : data_points) {
        double min_dist = std::numeric_limits<double>::max();
        int best_j = -1;
        for (int j = 0; j < k_; ++j) {
            double dist = compute_distance(point, centroids_.row(j).transpose());
            if (dist < min_dist) {
                min_dist = dist;
                best_j = j;
            }
        }
        predictions.push_back(best_j);
    }

    return predictions;
}

// Returns centroid matrix μ ∈ ℝ^{k × d}
Eigen::MatrixXd KMeans::get_centroids() const {
    return centroids_;
}

// Euclidean norm: ‖a - b‖₂
double KMeans::compute_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return (a - b).norm();
}

// μⱼ ← mean of points assigned to cluster j
void KMeans::update_centroids(const std::vector<Eigen::VectorXd>& points, const std::vector<int>& labels) {
    int d = points[0].size();
    std::vector<Eigen::VectorXd> sums(k_, Eigen::VectorXd::Zero(d));
    std::vector<int> counts(k_, 0);

    for (size_t i = 0; i < points.size(); ++i) {
        int label = labels[i];
        sums[label] += points[i];
        counts[label]++;
    }

    for (int j = 0; j < k_; ++j) {
        if (counts[j] > 0) {
            centroids_.row(j) = (sums[j] / counts[j]).transpose();
        }
    }
}
