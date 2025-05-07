#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

struct KeyPoint {
    cv::Point2f pt;    // Keypoint coordinates
    float scale;       // Scale at which keypoint was detected
    float response;    // Strength of the keypoint
};

class FeatureDetector {
public:
    FeatureDetector(float sigma = 1.6, int num_octaves = 4, int scales_per_octave = 3);
    void detect(const cv::Mat& image, std::vector<KeyPoint>& keypoints);

private:
    float sigma;
    int num_octaves;
    int scales_per_octave;

    void build_gaussian_pyramid(const cv::Mat& image, std::vector<std::vector<cv::Mat>>& pyramid);
    void compute_dog(const std::vector<std::vector<cv::Mat>>& gaussian_pyramid,
                     std::vector<std::vector<cv::Mat>>& dog_pyramid);
    void find_extrema(const std::vector<std::vector<cv::Mat>>& dog_pyramid,
                      std::vector<KeyPoint>& keypoints);
};

#endif // FEATURE_DETECTOR_H