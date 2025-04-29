#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class ContourDetector
{
public:
    ContourDetector();

    void process(const cv::Mat& input_image);
    void draw_contours(cv::Mat& output_image) const;

private:
    cv::Mat gray_image_;
    cv::Mat blurred_image_;
    cv::Mat edges_;
    std::vector<std::vector<cv::Point>> contours_;

    void convert_to_grayscale(const cv::Mat& input_image);
    void apply_gaussian_blur();
    void apply_manual_sobel();
    void apply_threshold();
    void find_contours();
};



