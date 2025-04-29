#pragma once

#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>

enum distance_metric {
    EUCLIDEAN,
    MANHATTAN,
    COSINE,
    CHEBYSHEV
};

enum normalization_type {
    NONE,
    MIN_MAX,
    Z_SCORE
};

class knn {
public:
    knn(int k, distance_metric metric = EUCLIDEAN, normalization_type norm_type = NONE);

    void train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);
    int predict(const cv::Mat& image);
    void set_distance_metric(distance_metric metric);
    void set_normalization_type(normalization_type norm_type);
    void set_logging(bool enable_logging);
    void display_summary();

private:
    int k_;
    distance_metric metric_;
    normalization_type norm_type_;
    std::vector<cv::Mat> train_images_;
    std::vector<int> train_labels_;
    bool logging_enabled_;

    double calculate_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b);
    Eigen::VectorXd extract_features(const cv::Mat& image);
    Eigen::VectorXd normalize_features(const Eigen::VectorXd& features);
    void log_message(const std::string& message);
    cv::Mat resize_image(const cv::Mat& image, const cv::Size& size);
    void calculate_normalization_stats(const std::vector<Eigen::VectorXd>& feature_vectors,
                                      Eigen::VectorXd& mean,
                                      Eigen::VectorXd& stddev);
};