#ifndef SIFT_H
#define SIFT_H

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <string>

class SIFT {
public:
    SIFT();

    void extractToCSV(const std::vector<cv::Mat>& imgs, const std::string& outCSV);

private:
    cv::Ptr<cv::xfeatures2d::SIFT> sift;  // SIFT detector and extractor

    void writeCSV(const std::vector<std::vector<cv::KeyPoint>>& kps, const std::vector<cv::Mat>& descs, const std::string& outCSV);

    std::vector<cv::Mat> loadImages(const std::vector<std::string>& paths);

    void logFeatures(const std::vector<std::vector<cv::KeyPoint>>& kps);
};

#endif // SIFT_H
