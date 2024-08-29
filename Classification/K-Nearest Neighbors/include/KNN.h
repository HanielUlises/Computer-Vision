#ifndef KNN_H
#define KNN_H

#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>

enum DistanceMetric {
    EUCLIDEAN,
    MANHATTAN,
    COSINE,
    CHEBYSHEV
};

enum NormalizationType {
    NONE,
    MIN_MAX,
    Z_SCORE
};

class KNN {
public:
    KNN(int k, DistanceMetric metric = EUCLIDEAN, NormalizationType normType = NONE);

    // Trains the KNN with provided images and labels
    void train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);

    // Predicts the label of a given image
    int predict(const cv::Mat& image);

    // Sets the distance metric to be used
    void setDistanceMetric(DistanceMetric metric);

    // Sets the normalization type for feature extraction
    void setNormalizationType(NormalizationType normType);

    // Enables or disables logging
    void setLogging(bool enableLogging);

    // Utility function to display a summary of the KNN model
    void displaySummary();

private:
    int k;  // Number of nearest neighbors
    DistanceMetric metric;  // Distance metric used
    NormalizationType normType;  // Normalization type used
    std::vector<cv::Mat> trainImages;  // Training images
    std::vector<int> trainLabels;  // Training labels
    bool loggingEnabled;  // Logging flag

    // Calculate the distance between two feature vectors
    double calculateDistance(const Eigen::VectorXd& a, const Eigen::VectorXd& b);

    // Extract features from an image, optionally normalizing them
    Eigen::VectorXd extractFeatures(const cv::Mat& image);

    // Normalize features using the selected normalization method
    Eigen::VectorXd normalizeFeatures(const Eigen::VectorXd& features);

    // Utility function to log a message
    void logMessage(const std::string& message);

    // Utility function to handle matrix resizing
    cv::Mat resizeImage(const cv::Mat& image, const cv::Size& size);

    // Calculate statistics (mean, standard deviation) for Z-score normalization
    void calculateNormalizationStats(const std::vector<Eigen::VectorXd>& featureVectors,
                                     Eigen::VectorXd& mean,
                                     Eigen::VectorXd& stddev);
};

#endif // KNN_H
