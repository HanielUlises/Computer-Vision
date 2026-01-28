#include "Steerable_Fourier.hpp"
#include <cmath>

SteerableFourier::SteerableFourier(
    int kernelSize,
    double s,
    double f
)
    : ksize(kernelSize), sigma(s), freq(f)
{
    buildKernels();
}

void SteerableFourier::buildKernels()
{
    baseX = cv::Mat(ksize, ksize, CV_64F);
    baseY = cv::Mat(ksize, ksize, CV_64F);

    int half = ksize / 2;

    for (int y = -half; y <= half; ++y)
    {
        for (int x = -half; x <= half; ++x)
        {
            double g =
                std::exp(-(x*x + y*y) / (2.0 * sigma * sigma));

            double cx = g * std::cos(2.0 * M_PI * freq * x);
            double cy = g * std::cos(2.0 * M_PI * freq * y);

            baseX.at<double>(y + half, x + half) = cx;
            baseY.at<double>(y + half, x + half) = cy;
        }
    }
}

void SteerableFourier::apply(
    const cv::Mat& src,
    cv::Mat& dst,
    double theta
) const
{
    cv::Mat respX, respY;

    cv::filter2D(src, respX, CV_64F, baseX);
    cv::filter2D(src, respY, CV_64F, baseY);

    dst = std::cos(theta) * respX + std::sin(theta) * respY;
}
