#ifndef KNN_H
#define KNN_H

#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>

class KNN {
public:
    KNN(int k);
    void train(const std::vector<cv::Mat>& images, const std::vector<int>& labels);
    int predict(const cv::Mat& image);

private:
    int k;
    std::vector<cv::Mat> trainImages;
    std::vector<int> trainLabels;

    double calculateDistance(const cv::Mat& a, const cv::Mat& b);
};

#endif // KNN_H