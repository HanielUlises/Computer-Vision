#pragma once

#include <opencv2/core.hpp>

class Shannon {
public:
    // Normalized frequency range (0.0 - 0.5)
    Shannon(double f_low, double f_high);

    void apply(const cv::Mat& src, cv::Mat& dst) const;

private:
    double f_low_;
    double f_high_;

    void createBandPassMask(cv::Mat& mask) const;
    static void fftShift(cv::Mat& img);
};