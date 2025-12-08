#ifndef SVM_H
#define SVM_H

#include <vector>
#include <map>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include <Eigen/Dense>

enum kernel_type {
    KERNEL_LINEAR,
    KERNEL_POLY,
    KERNEL_RBF,
    KERNEL_SIGMOID
};

struct svm_parameters {
    kernel_type kernel;
    double C;
    double gamma;
    int degree;
    double coef0;
};

class svm_classifier {
public:
    svm_classifier(const svm_parameters& params);

    void train(const std::vector<cv::Mat>& images,
               const std::vector<int>& labels);

    int predict(const cv::Mat& image);

    std::map<std::string, double> evaluate(const std::vector<cv::Mat>& images,
                                           const std::vector<int>& labels);

    void set_logging(bool enable);

    void summary();

private:
    svm_parameters params_;
    cv::Ptr<cv::ml::SVM> model_;
    bool logging_enabled_;

    Eigen::VectorXd extract_features(const cv::Mat& image);
    Eigen::VectorXd normalize_features(const Eigen::VectorXd& vec);

    cv::Mat eigen_to_mat_row(const Eigen::VectorXd& vec);
    Eigen::VectorXd mat_to_eigen_row(const cv::Mat& mat);

    void log_message(const std::string& msg);

    void compute_confusion_matrix(const std::vector<int>& y_true,
                                  const std::vector<int>& y_pred,
                                  int num_classes);

    void compute_precision_recall_f1(const std::vector<int>& y_true,
                                     const std::vector<int>& y_pred,
                                     int num_classes);
};

#endif // SVM_H
