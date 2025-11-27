#include "KNN.hpp"
#include <limits>
#include <algorithm>
#include <numeric>

knn::knn(int k, distance_metric metric, normalization_type norm_type)
    : k_(k), metric_(metric), norm_type_(norm_type), logging_enabled_(false) {}

void knn::train(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
    train_images_ = images;
    train_labels_ = labels;
    log_message("Training completed with " + std::to_string(images.size()) + " images.");
}

int knn::predict(const cv::Mat& image) {
    std::vector<std::pair<double, int>> distances;
    Eigen::VectorXd query_features = extract_features(image);

    for (size_t i = 0; i < train_images_.size(); i++) {
        Eigen::VectorXd train_features = extract_features(train_images_[i]);
        double dist = calculate_distance(train_features, query_features);
        distances.emplace_back(dist, train_labels_[i]);
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> k_votes(k_);
    for (int i = 0; i < k_; ++i) {
        k_votes[i] = distances[i].second;
    }

    std::sort(k_votes.begin(), k_votes.end());
    int most_common = k_votes[0], max_count = 1, current_count = 1;
    for (int i = 1; i < k_; ++i) {
        if (k_votes[i] == k_votes[i - 1]) {
            current_count++;
            if (current_count > max_count) {
                max_count = current_count;
                most_common = k_votes[i];
            }
        } else {
            current_count = 1;
        }
    }

    log_message("Predicted class: " + std::to_string(most_common));
    return most_common;
}

void knn::set_distance_metric(distance_metric metric) {
    metric_ = metric;
    log_message("Distance metric set.");
}

void knn::set_normalization_type(normalization_type norm_type) {
    norm_type_ = norm_type;
    log_message("Normalization type set.");
}

void knn::set_logging(bool enable_logging) {
    logging_enabled_ = enable_logging;
}

void knn::log_message(const std::string& message) {
    if (logging_enabled_) {
        std::cout << "[KNN Log]: " << message << std::endl;
    }
}

void knn::display_summary() {
    std::cout << "KNN Model Summary:" << std::endl;
    std::cout << "K value: " << k_ << std::endl;
    std::cout << "Distance Metric: ";
    switch (metric_) {
        case EUCLIDEAN: std::cout << "Euclidean"; break;
        case MANHATTAN: std::cout << "Manhattan"; break;
        case COSINE: std::cout << "Cosine"; break;
        case CHEBYSHEV: std::cout << "Chebyshev"; break;
    }
    std::cout << std::endl;

    std::cout << "Normalization Type: ";
    switch (norm_type_) {
        case NONE: std::cout << "None"; break;
        case MIN_MAX: std::cout << "Min-Max"; break;
        case Z_SCORE: std::cout << "Z-Score"; break;
    }
    std::cout << std::endl;
    std::cout << "Number of training samples: " << train_images_.size() << std::endl;
}

double knn::calculate_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    switch (metric_) {
        case EUCLIDEAN:
            return (a - b).norm();
        case MANHATTAN:
            return (a - b).cwiseAbs().sum();
        case COSINE:
            return 1.0 - (a.dot(b) / (a.norm() * b.norm()));
        case CHEBYSHEV:
            return (a - b).cwiseAbs().maxCoeff();
        default:
            throw std::invalid_argument("Unknown distance metric.");
    }
}

Eigen::VectorXd knn::extract_features(const cv::Mat& image) {
    cv::Mat resized = resize_image(image, cv::Size(32, 32));
    Eigen::VectorXd features(resized.rows * resized.cols);

    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            features(i * resized.cols + j) = resized.at<uchar>(i, j);
        }
    }

    return normalize_features(features);
}

Eigen::VectorXd knn::normalize_features(const Eigen::VectorXd& features) {
    if (norm_type_ == NONE) {
        return features;
    } else if (norm_type_ == MIN_MAX) {
        double min_val = features.minCoeff();
        double max_val = features.maxCoeff();
        return (features.array() - min_val) / (max_val - min_val);
    } else if (norm_type_ == Z_SCORE) {
        Eigen::VectorXd mean, stddev;
        calculate_normalization_stats({features}, mean, stddev);
        return (features.array() - mean.array()) / stddev.array();
    } else {
        throw std::invalid_argument("Unknown normalization type.");
    }
}

cv::Mat knn::resize_image(const cv::Mat& image, const cv::Size& size) {
    cv::Mat resized;
    cv::resize(image, resized, size);
    return resized;
}

void knn::calculate_normalization_stats(const std::vector<Eigen::VectorXd>& feature_vectors,
                                       Eigen::VectorXd& mean,
                                       Eigen::VectorXd& stddev) {
    if (feature_vectors.empty()) return;

    int num_features = feature_vectors[0].size();
    mean = Eigen::VectorXd::Zero(num_features);
    stddev = Eigen::VectorXd::Zero(num_features);

    for (const auto& vec : feature_vectors) {
        mean += vec;
    }
    mean /= static_cast<double>(feature_vectors.size());

    for (const auto& vec : feature_vectors) {
        stddev += (vec - mean).cwiseAbs2();
    }
    stddev = (stddev / static_cast<double>(feature_vectors.size())).cwiseSqrt();
}