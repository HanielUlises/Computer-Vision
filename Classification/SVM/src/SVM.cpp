#include "SVM.h"

svm_classifier::svm_classifier(const svm_parameters& params)
    : params_(params), logging_enabled_(false) {
    model_ = cv::ml::SVM::create();
    model_->setType(cv::ml::SVM::C_SVC);
    model_->setC(params_.C);

    switch (params_.kernel) {
        case KERNEL_LINEAR:  model_->setKernel(cv::ml::SVM::LINEAR); break;
        case KERNEL_POLY:    model_->setKernel(cv::ml::SVM::POLY); break;
        case KERNEL_RBF:     model_->setKernel(cv::ml::SVM::RBF); break;
        case KERNEL_SIGMOID: model_->setKernel(cv::ml::SVM::SIGMOID); break;
    }

    model_->setGamma(params_.gamma);
    model_->setDegree(params_.degree);
    model_->setCoef0(params_.coef0);
}

void svm_classifier::set_logging(bool enable) {
    logging_enabled_ = enable;
}

void svm_classifier::log_message(const std::string& msg) {
    if (logging_enabled_) {
        std::cout << "[SVM] " << msg << std::endl;
    }
}

/*
-------------------------------------------------------------
   EIGEN <-> OPENCV CONVERSION HELPERS
-------------------------------------------------------------
*/

cv::Mat svm_classifier::eigen_to_mat_row(const Eigen::VectorXd& vec) {
    cv::Mat mat(1, vec.size(), CV_32F);
    for (int i = 0; i < vec.size(); i++)
        mat.at<float>(0, i) = static_cast<float>(vec(i));
    return mat;
}

Eigen::VectorXd svm_classifier::mat_to_eigen_row(const cv::Mat& mat) {
    Eigen::VectorXd vec(mat.cols);
    for (int i = 0; i < mat.cols; i++)
        vec(i) = mat.at<float>(0, i);
    return vec;
}

/*
-------------------------------------------------------------
   FEATURE EXTRACTION (EIGEN)
-------------------------------------------------------------
*/

Eigen::VectorXd svm_classifier::extract_features(const cv::Mat& image) {
    cv::Mat gray, resized;

    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.clone();

    cv::resize(gray, resized, cv::Size(64, 64));  // Feature size
    resized.convertTo(resized, CV_32F);

    Eigen::VectorXd vec(64 * 64);

    for (int r = 0; r < 64; r++)
        for (int c = 0; c < 64; c++)
            vec(r * 64 + c) = resized.at<float>(r, c);

    return normalize_features(vec);
}

Eigen::VectorXd svm_classifier::normalize_features(const Eigen::VectorXd& vec) {
    double min_val = vec.minCoeff();
    double max_val = vec.maxCoeff();

    if (max_val - min_val < 1e-12)
        return vec;

    return (vec.array() - min_val) / (max_val - min_val);
}

void svm_classifier::train(const std::vector<cv::Mat>& images,
                           const std::vector<int>& labels) {
    log_message("Extracting features...");

    cv::Mat training_data;
    cv::Mat label_mat(labels.size(), 1, CV_32S);

    for (size_t i = 0; i < images.size(); i++) {
        Eigen::VectorXd feat = extract_features(images[i]);
        cv::Mat feat_row = eigen_to_mat_row(feat);

        training_data.push_back(feat_row);
        label_mat.at<int>(i, 0) = labels[i];
    }

    log_message("Training SVM...");
    model_->train(training_data, cv::ml::ROW_SAMPLE, label_mat);
    log_message("Training completed.");
}

int svm_classifier::predict(const cv::Mat& image) {
    Eigen::VectorXd feat = extract_features(image);
    cv::Mat feat_row = eigen_to_mat_row(feat);
    return static_cast<int>(model_->predict(feat_row));
}

std::map<std::string, double> svm_classifier::evaluate(
        const std::vector<cv::Mat>& images,
        const std::vector<int>& labels) {

    std::vector<int> preds(labels.size());

    for (size_t i = 0; i < images.size(); i++)
        preds[i] = predict(images[i]);

    int correct = 0;
    for (size_t i = 0; i < labels.size(); i++)
        if (preds[i] == labels[i])
            correct++;

    double accuracy = correct / double(labels.size());

    log_message("Evaluation completed.");
    std::cout << "Accuracy: " << accuracy << std::endl;

    compute_confusion_matrix(labels, preds, 10);
    compute_precision_recall_f1(labels, preds, 10);

    return { {"accuracy", accuracy} };
}

void svm_classifier::compute_confusion_matrix(
        const std::vector<int>& y_true,
        const std::vector<int>& y_pred,
        int num_classes) {
    Eigen::MatrixXi cm = Eigen::MatrixXi::Zero(num_classes, num_classes);

    for (size_t i = 0; i < y_true.size(); i++)
        cm(y_true[i], y_pred[i])++;

    std::cout << "\nConfusion Matrix:\n" << cm << "\n";
}


void svm_classifier::compute_precision_recall_f1(
        const std::vector<int>& y_true,
        const std::vector<int>& y_pred,
        int num_classes) {
    Eigen::VectorXi tp = Eigen::VectorXi::Zero(num_classes);
    Eigen::VectorXi fp = Eigen::VectorXi::Zero(num_classes);
    Eigen::VectorXi fn = Eigen::VectorXi::Zero(num_classes);

    for (size_t i = 0; i < y_true.size(); i++) {
        int t = y_true[i];
        int p = y_pred[i];

        if (t == p)
            tp(t)++;
        else {
            fp(p)++;
            fn(t)++;
        }
    }

    std::cout << "\nPrecision / Recall / F1 per class:\n";

    for (int c = 0; c < num_classes; c++) {
        double precision = (tp(c) + fp(c)) ? tp(c) / double(tp(c) + fp(c)) : 0.0;
        double recall    = (tp(c) + fn(c)) ? tp(c) / double(tp(c) + fn(c)) : 0.0;
        double f1 = (precision + recall > 0) ?
                      2.0 * precision * recall / (precision + recall) : 0.0;

        std::cout << "Class " << c
                  << " | Precision: " << precision
                  << " | Recall: "    << recall
                  << " | F1: "        << f1 << "\n";
    }
}


void svm_classifier::summary() {
    std::cout << "=== SVM Model Summary ===\n";
    std::cout << "Kernel: " << params_.kernel << "\n";
    std::cout << "C: " << params_.C << "\n";
    std::cout << "Gamma: " << params_.gamma << "\n";
    std::cout << "Degree: " << params_.degree << "\n";    std::cout << "Coef0: " << params_.coef0 << "\n";
}
