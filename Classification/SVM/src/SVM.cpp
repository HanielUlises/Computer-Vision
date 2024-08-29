#include "SVM.h"
#include <opencv2/ml/ml.hpp>
#include <iomanip>
#include <fstream>

// Constructor: Initialize the SVM with given parameters
SVM::SVM(const SVMParameters& params) : params(params), loggingEnabled(false) {
    model = cv::ml::SVM::create();
    model->setC(params.C);
    model->setKernel(params.kernelType == LINEAR ? cv::ml::SVM::LINEAR :
                     params.kernelType == POLY ? cv::ml::SVM::POLY :
                     params.kernelType == RBF ? cv::ml::SVM::RBF :
                     cv::ml::SVM::SIGMOID);
    model->setGamma(params.gamma);
    model->setDegree(params.degree);
    model->setCoef0(params.coef0);
}

// Train the SVM model with provided images and labels
void SVM::train(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
    cv::Mat trainingData(images.size(), images[0].rows * images[0].cols, CV_32F);
    cv::Mat trainingLabels(labels.size(), 1, CV_32S);

    for (size_t i = 0; i < images.size(); ++i) {
        Eigen::VectorXd features = extractFeatures(images[i]);
        Eigen::VectorXd normalizedFeatures = normalizeFeatures(features);
        for (int j = 0; j < features.size(); ++j) {
            trainingData.at<float>(i, j) = static_cast<float>(normalizedFeatures(j));
        }
        trainingLabels.at<int>(i, 0) = labels[i];
    }

    model->train(trainingData, cv::ml::ROW_SAMPLE, trainingLabels);

    logMessage("Training completed.");
}

// Predict the class of a given image
int SVM::predict(const cv::Mat& image) {
    Eigen::VectorXd features = extractFeatures(image);
    Eigen::VectorXd normalizedFeatures = normalizeFeatures(features);
    cv::Mat sample(1, features.size(), CV_32F);

    for (int i = 0; i < features.size(); ++i) {
        sample.at<float>(0, i) = static_cast<float>(normalizedFeatures(i));
    }

    int prediction = static_cast<int>(model->predict(sample));
    logMessage("Prediction: " + std::to_string(prediction));
    return prediction;
}

// Evaluate the model on a test set
std::map<std::string, double> SVM::evaluate(const std::vector<cv::Mat>& testImages, const std::vector<int>& testLabels) {
    std::vector<int> predictedLabels;
    std::vector<double> decisionValues;

    for (const auto& image : testImages) {
        Eigen::VectorXd features = extractFeatures(image);
        Eigen::VectorXd normalizedFeatures = normalizeFeatures(features);
        cv::Mat sample(1, features.size(), CV_32F);

        for (int i = 0; i < features.size(); ++i) {
            sample.at<float>(0, i) = static_cast<float>(normalizedFeatures(i));
        }

        predictedLabels.push_back(static_cast<int>(model->predict(sample, decisionValues)));
    }

    // Statistical analysis
    calculateConfusionMatrix(testLabels, predictedLabels, *std::max_element(testLabels.begin(), testLabels.end()) + 1);
    calculatePrecisionRecallF1(testLabels, predictedLabels, *std::max_element(testLabels.begin(), testLabels.end()) + 1);
    if (params.kernelType == LINEAR && testLabels.size() == 2) {
        generateROCCurve(testLabels, decisionValues);
    }

    // Returning some metrics as a map
    std::map<std::string, double> metrics;
    metrics["Accuracy"] = static_cast<double>(std::count(predictedLabels.begin(), predictedLabels.end(), testLabels) / testLabels.size());
    logMessage("Evaluation completed.");
    return metrics;
}

// Convert an image to a feature vector
Eigen::VectorXd SVM::extractFeatures(const cv::Mat& image) {
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(32, 32));  // Resize for consistency
    Eigen::VectorXd features(resized.rows * resized.cols);

    for (int i = 0; i < resized.rows; ++i) {
        for (int j = 0; j < resized.cols; ++j) {
            features(i * resized.cols + j) = static_cast<double>(resized.at<uchar>(i, j));
        }
    }

    return features;
}

// Normalize features to range [0, 1]
Eigen::VectorXd SVM::normalizeFeatures(const Eigen::VectorXd& features) {
    double minVal = features.minCoeff();
    double maxVal = features.maxCoeff();
    return (features.array() - minVal) / (maxVal - minVal);
}

// Calculate and print confusion matrix
void SVM::calculateConfusionMatrix(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels, int numClasses) {
    Eigen::MatrixXi confusionMatrix = Eigen::MatrixXi::Zero(numClasses, numClasses);

    for (size_t i = 0; i < trueLabels.size(); ++i) {
        confusionMatrix(trueLabels[i], predictedLabels[i])++;
    }

    std::cout << "Confusion Matrix:\n" << confusionMatrix << std::endl;
}

// Calculate and print precision, recall, and F1-score
void SVM::calculatePrecisionRecallF1(const std::vector<int>& trueLabels, const std::vector<int>& predictedLabels, int numClasses) {
    std::vector<int> tp(numClasses, 0), fp(numClasses, 0), fn(numClasses, 0);

    for (size_t i = 0; i < trueLabels.size(); ++i) {
        if (trueLabels[i] == predictedLabels[i]) {
            tp[trueLabels[i]]++;
        } else {
            fp[predictedLabels[i]]++;
            fn[trueLabels[i]]++;
        }
    }

    for (int i = 0; i < numClasses; ++i) {
        double precision = tp[i] / static_cast<double>(tp[i] + fp[i]);
        double recall = tp[i] / static_cast<double>(tp[i] + fn[i]);
        double f1 = 2 * (precision * recall) / (precision + recall);
        std::cout << "Class " << i << " - Precision: " << precision << ", Recall: " << recall << ", F1-Score: " << f1 << std::endl;
    }
}

// Generate and save ROC curves for binary classification
void SVM::generateROCCurve(const std::vector<int>& trueLabels, const std::vector<double>& decisionValues) {
    std::ofstream rocFile("roc_curve.csv");
    rocFile << "Threshold,TPR,FPR\n";

    std::vector<double> sortedDecisionValues = decisionValues;
    std::sort(sortedDecisionValues.begin(), sortedDecisionValues.end());

    for (double threshold : sortedDecisionValues) {
        int tp = 0, fp = 0, tn = 0, fn = 0;

        for (size_t i = 0; i < trueLabels.size(); ++i) {
            if (decisionValues[i] >= threshold) {
                if (trueLabels[i] == 1) tp++;
                else fp++;
            } else {
                if (trueLabels[i] == 0) tn++;
                else fn++;
            }
        }

        double tpr = tp / static_cast<double>(tp + fn);
        double fpr = fp / static_cast<double>(fp + tn);
        rocFile << threshold << "," << tpr << "," << fpr << "\n";
    }

    rocFile.close();
    std::cout << "ROC curve saved to 'roc_curve.csv'." << std::endl;
}

// Log a message if logging is enabled
void SVM::logMessage(const std::string& message) {
    if (loggingEnabled) {
        std::cout << message << std::endl;
    }
}

// Enable or disable logging
void SVM::setLogging(bool enableLogging) {
    loggingEnabled = enableLogging;
}

void SVM::displaySummary() {
    std::cout << "SVM Model Summary:" << std::endl;
    std::cout << "Kernel: " << (params.kernelType == LINEAR ? "Linear" :
                                params.kernelType == POLY ? "Polynomial" :
                                params.kernelType == RBF ? "RBF" : "Sigmoid") << std::endl;
    std::cout << "C: " << params.C << std::endl;
    std::cout << "Gamma: " << params.gamma << std::endl;
    std::cout << "Degree (for Poly kernel): " << params.degree << std::endl;
    std::cout << "Coef0 (for Poly/Sigmoid kernel): " << params.coef0 << std::endl;
}
