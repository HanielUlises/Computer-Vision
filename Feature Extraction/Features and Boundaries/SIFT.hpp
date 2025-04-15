#ifndef SIFT_H
#define SIFT_H

#include <opencv2/core.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <string>

class sift {
public:
    sift();
    void extract_to_csv(const std::vector<cv::Mat>& images, const std::string& output_csv);
    std::vector<cv::Mat> load_images(const std::vector<std::string>& paths);
    void detect_and_compute(const cv::Mat& image, 
                           std::vector<cv::KeyPoint>& keypoints, 
                           cv::Mat& descriptors);
    void build_vocabulary(const std::vector<cv::Mat>& images, int cluster_count);
    std::vector<cv::Mat> compute_bow_histograms(const std::vector<cv::Mat>& images);

private:
    cv::Ptr<cv::xfeatures2d::SIFT> sift_detector;
    cv::Mat vocabulary; // Cluster centroids from KMeans
    void write_csv(const std::vector<std::vector<cv::KeyPoint>>& keypoints, 
                   const std::vector<cv::Mat>& descriptors, 
                   const std::string& output_csv);
    void log_features(const std::vector<std::vector<cv::KeyPoint>>& keypoints);
};

#endif