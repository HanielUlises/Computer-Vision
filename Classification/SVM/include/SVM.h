#ifndef SVM_H
#define SVM_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <map>
#include <string>
#include <tuple>

enum KernelType {
    LINEAR,
    POLY,
    RBF,
    SIGMOID
};

struct SVMParameters {
    KernelType kernelType;
    double C;
    double gamma;
    int degree;
    double coef0;
};

class SVM {
public:
    SVM(const SVMParameters& params);

    // Train the SVM model
    void train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);

    // Predict the class of a given image
    int predict(const cv::Mat& image);

    // Evaluate the model on a test set
    std::map<std::string, double> evaluate(const std::vector<cv::Mat>& testImages, const std::vector<int>& testLabels);

    // Enable or disable logging
    void setLogging(bool enableLogging);

    // Display a summary of the model and its parameters
    void displaySummary();

private:
    SVMParameters params;  // SVM parameters
    cv::Ptr<cv::ml::SVM> model;  // OpenCV's SVM model
    bool loggingEnabled;  // Logging flag

    // Convert an image to a feature vector
    Eigen::VectorXd extractFeatures(const cv::Mat& image);

    // Log a message if logging is enabled
    void logMessage(const std::string& message);

    // Calculate and print confusion matrix
    void calculateConfusionMatrix(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels, int numClasses);

    // Calculate and print precision, recall, and F1-score
    void calculatePrecisionRecallF1(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels, int numClasses);

    // Generate and save ROC curves for binary classification
    void generateROCCurve(const std::vector<int>& trueLabels, const std::vector<double>& decisionValues);

    // Normalize features for consistency
    Eigen::VectorXd normalizeFeatures(const Eigen::VectorXd& features);
};

#endif // SVM_H
