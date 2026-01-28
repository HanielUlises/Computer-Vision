#pragma once
#include <opencv2/opencv.hpp>

class SteerableFourier
{
public:
    SteerableFourier(
        int kernelSize,
        double sigma,
        double frequency
    );

    void apply(
        const cv::Mat& src,
        cv::Mat& dst,
        double theta
    ) const;

private:
    int ksize;
    double sigma;
    double freq;

    cv::Mat baseX;
    cv::Mat baseY;

    void buildKernels();
};
