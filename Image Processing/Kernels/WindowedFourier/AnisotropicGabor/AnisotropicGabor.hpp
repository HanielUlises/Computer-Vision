#pragma once
#include <opencv2/opencv.hpp>

class AnisotropicGabor
{
public:
    AnisotropicGabor(
        int kernelSize,
        double sigmaX,
        double sigmaY,
        double wavelength,
        double orientation,
        double phase = 0.0
    );

    void apply(
        const cv::Mat& src,
        cv::Mat& dst
    ) const;

    const cv::Mat& kernel() const;

private:
    int ksize;
    double sx;
    double sy;
    double lambda;
    double theta;
    double phi;

    cv::Mat kernelMat;

    void buildKernel();
};
