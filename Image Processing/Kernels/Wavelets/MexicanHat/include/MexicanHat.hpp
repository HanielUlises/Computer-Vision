#pragma once

#include <opencv2/core.hpp>

class MexicanHat {
public:
    // sigma controls scale
    // ksize must be odd
    MexicanHat(int ksize, double sigma);

    void apply(const cv::Mat& src, cv::Mat& dst) const;

private:
    int ksize_;
    double sigma_;

    cv::Mat kernel_;

    void createKernel();
};