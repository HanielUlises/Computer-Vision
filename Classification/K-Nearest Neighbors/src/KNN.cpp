#include "KNN.h"
#include <limits>
#include <algorithm>
#include <numeric>

// Constructor: Initializes KNN with specified parameters
KNN::KNN(int k, DistanceMetric metric, NormalizationType normType)
    : k(k), metric(metric), normType(normType), loggingEnabled(false) {}

// Trains the KNN model with given training images and labels
void KNN::train(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
    trainImages = images;
    trainLabels = labels;
    
    logMessage("Training completed with " + std::to_string(images.size()) + " images.");
}

// Predicts the class label for a given test image
int KNN::predict(const cv::Mat& image) {
    std::vector<std::pair<double, int>> distances;
    Eigen::VectorXd queryFeatures = extractFeatures(image);

    for (size_t i = 0; i < trainImages.size(); i++) {
        Eigen::VectorXd trainFeatures = extractFeatures(trainImages[i]);
        double dist = calculateDistance(trainFeatures, queryFeatures);
        distances.emplace_back(dist, trainLabels[i]);
    }

    std::sort(distances.begin(), distances.end());

    std::vector<int> kVotes(k);
    for (int i = 0; i < k; ++i) {
        kVotes[i] = distances[i].second;
    }

    std::sort(kVotes.begin(), kVotes.end());
    int mostCommon = kVotes[0], maxCount = 1, currentCount = 1;
    for (int i = 1; i < k; ++i) {
        if (kVotes[i] == kVotes[i - 1]) {
            currentCount++;
            if (currentCount > maxCount) {
                maxCount = currentCount;
                mostCommon = kVotes[i];
            }
        } else {
            currentCount = 1;
        }
    }

    logMessage("Predicted class: " + std::to_string(mostCommon));
    return mostCommon;
}

// Sets the distance metric to be used for predictions
void KNN::setDistanceMetric(DistanceMetric metric) {
    this->metric = metric;
    logMessage("Distance metric set.");
}

// Sets the normalization type for feature extraction
void KNN::setNormalizationType(NormalizationType normType) {
    this->normType = normType;
    logMessage("Normalization type set.");
}

// Enables or disables logging for the KNN class
void KNN::setLogging(bool enableLogging) {
    loggingEnabled = enableLogging;
}

// Logs a message if logging is enabled
void KNN::logMessage(const std::string& message) {
    if (loggingEnabled) {
        std::cout << "[KNN Log]: " << message << std::endl;
    }
}

// Displays a summary of the current KNN model's configuration
void KNN::displaySummary() {
    std::cout << "KNN Model Summary:" << std::endl;
    std::cout << "K value: " << k << std::endl;
    std::cout << "Distance Metric: ";
    switch (metric) {
        case EUCLIDEAN: std::cout << "Euclidean"; break;
        case MANHATTAN: std::cout << "Manhattan"; break;
        case COSINE: std::cout << "Cosine"; break;
        case CHEBYSHEV: std::cout << "Chebyshev"; break;
    }
    std::cout << std::endl;

    std::cout << "Normalization Type: ";
    switch (normType) {
        case NONE: std::cout << "None"; break;
        case MIN_MAX: std::cout << "Min-Max"; break;
        case Z_SCORE: std::cout << "Z-Score"; break;
    }
    std::cout << std::endl;
    std::cout << "Number of training samples: " << trainImages.size() << std::endl;
}

// Calculates the distance between two feature vectors using the selected metric
double KNN::calculateDistance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    switch (metric) {
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

// Extracts features from an image, normalizes them if required
Eigen::VectorXd KNN::extractFeatures(const cv::Mat& image) {
    cv::Mat resized = resizeImage(image, cv::Size(32, 32));
    Eigen::VectorXd features(resized.rows * resized.cols);

    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            features(i * resized.cols + j) = resized.at<uchar>(i, j);
        }
    }

    return normalizeFeatures(features);
}

// Normalizes the features using the selected normalization method
Eigen::VectorXd KNN::normalizeFeatures(const Eigen::VectorXd& features) {
    if (normType == NONE) {
        return features;
    } else if (normType == MIN_MAX) {
        double minVal = features.minCoeff();
        double maxVal = features.maxCoeff();
        return (features.array() - minVal) / (maxVal - minVal);
    } else if (normType == Z_SCORE) {
        Eigen::VectorXd mean, stddev;
        calculateNormalizationStats({features}, mean, stddev);
        return (features.array() - mean.array()) / stddev.array();
    } else {
        throw std::invalid_argument("Unknown normalization type.");
    }
}

// Resizes an image to a specified size
cv::Mat KNN::resizeImage(const cv::Mat& image, const cv::Size& size) {
    cv::Mat resized;
    cv::resize(image, resized, size);
    return resized;
}

// Calculates mean and standard deviation for Z-score normalization
void KNN::calculateNormalizationStats(const std::vector<Eigen::VectorXd>& featureVectors,
                                      Eigen::VectorXd& mean,
                                      Eigen::VectorXd& stddev) {
    if (featureVectors.empty()) return;

    int numFeatures = featureVectors[0].size();
    mean = Eigen::VectorXd::Zero(numFeatures);
    stddev = Eigen::VectorXd::Zero(numFeatures);

    // Mean
    for (const auto& vec : featureVectors) {
        mean += vec;
    }
    mean /= static_cast<double>(featureVectors.size());

    // Standard deviation
    for (const auto& vec : featureVectors) {
        stddev += (vec - mean).cwiseAbs2();
    }
    stddev = (stddev / static_cast<double>(featureVectors.size())).cwiseSqrt();
}