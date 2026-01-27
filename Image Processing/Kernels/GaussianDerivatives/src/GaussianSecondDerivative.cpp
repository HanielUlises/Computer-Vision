#include "GaussianSecondDerivative.hpp"
#include <cmath>


static inline double G(double x, double y, double sigma2)
{
    return std::exp(-(x*x + y*y) / (2.0 * sigma2));
}

cv::Mat GaussianSecondDerivative::dxx(const Params& p)
{
    CV_Assert(p.ksize % 2 == 1);
    int h = p.ksize / 2;
    double sigma2 = p.sigma * p.sigma;
    double sigma4 = sigma2 * sigma2;

    cv::Mat k(p.ksize, p.ksize, CV_32F);

    for (int y = -h; y <= h; ++y)
    for (int x = -h; x <= h; ++x)
    {
        double val = (x*x - sigma2) / sigma4 * G(x, y, sigma2);
        k.at<float>(y+h, x+h) = static_cast<float>(val);
    }

    return k;
}

cv::Mat GaussianSecondDerivative::dyy(const Params& p)
{
    CV_Assert(p.ksize % 2 == 1);
    int h = p.ksize / 2;
    double sigma2 = p.sigma * p.sigma;
    double sigma4 = sigma2 * sigma2;

    cv::Mat k(p.ksize, p.ksize, CV_32F);

    for (int y = -h; y <= h; ++y)
    for (int x = -h; x <= h; ++x)
    {
        double val = (y*y - sigma2) / sigma4 * G(x, y, sigma2);
        k.at<float>(y+h, x+h) = static_cast<float>(val);
    }

    return k;
}

cv::Mat GaussianSecondDerivative::dxy(const Params& p)
{
    CV_Assert(p.ksize % 2 == 1);
    int h = p.ksize / 2;
    double sigma2 = p.sigma * p.sigma;
    double sigma4 = sigma2 * sigma2;

    cv::Mat k(p.ksize, p.ksize, CV_32F);

    for (int y = -h; y <= h; ++y)
    for (int x = -h; x <= h; ++x)
    {
        double val = (x * y) / sigma4 * G(x, y, sigma2);
        k.at<float>(y+h, x+h) = static_cast<float>(val);
    }

    return k;
}