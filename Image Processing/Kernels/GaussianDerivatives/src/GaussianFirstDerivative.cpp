#include "GaussianFirstDerivative.hpp"
#include <cmath>

cv::Mat GaussianFirstDerivative::create(const Params& p)
{
    CV_Assert(p.ksize % 2 == 1);
    int h = p.ksize / 2;

    cv::Mat k(p.ksize, p.ksize, CV_32F);

    double ct = std::cos(p.theta);
    double st = std::sin(p.theta);
    double sigma2 = p.sigma * p.sigma;

    for (int y = -h; y <= h; ++y)
    for (int x = -h; x <= h; ++x)
    {
        double xd = x * ct + y * st;
        double gauss =
            std::exp(-(x*x + y*y) / (2.0 * sigma2));

        double val = -xd / sigma2 * gauss;

        k.at<float>(y + h, x + h) = static_cast<float>(val);
    }

    return k;
}