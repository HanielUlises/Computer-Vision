#include "Morlet.hpp"
#include <cmath>


static inline void rotatedCoords(
    int x, int y, double theta,
    double& xr, double& yr)
{
    double ct = std::cos(theta);
    double st = std::sin(theta);
    xr =  x * ct + y * st;
    yr = -x * st + y * ct;
}

cv::Mat MorletWavelet::real(const Params& p)
{
    CV_Assert(p.ksize % 2 == 1);
    int h = p.ksize / 2;

    cv::Mat k(p.ksize, p.ksize, CV_32F);

    const double sigma2 = p.sigma * p.sigma;
    const double k0 = 2.0 * CV_PI * p.frequency;

    // DC correction term (admissibility)
    const double C = std::exp(-0.5 * sigma2 * k0 * k0);

    for (int y = -h; y <= h; ++y)
    for (int x = -h; x <= h; ++x)
    {
        double xr, yr;
        rotatedCoords(x, y, p.theta, xr, yr);

        double gauss = std::exp(-(x*x + y*y) / (2.0 * sigma2));
        double wave  = std::cos(k0 * xr) - C;

        k.at<float>(y + h, x + h) =
            static_cast<float>(gauss * wave);
    }

    return k;
}

cv::Mat MorletWavelet::imaginary(const Params& p)
{
    CV_Assert(p.ksize % 2 == 1);
    int h = p.ksize / 2;

    cv::Mat k(p.ksize, p.ksize, CV_32F);

    const double sigma2 = p.sigma * p.sigma;
    const double k0 = 2.0 * CV_PI * p.frequency;

    for (int y = -h; y <= h; ++y)
    for (int x = -h; x <= h; ++x)
    {
        double xr, yr;
        rotatedCoords(x, y, p.theta, xr, yr);

        double gauss = std::exp(-(x*x + y*y) / (2.0 * sigma2));
        double wave  = std::sin(k0 * xr);

        k.at<float>(y + h, x + h) =
            static_cast<float>(gauss * wave);
    }

    return k;
}

std::pair<cv::Mat, cv::Mat>
MorletWavelet::complex(const Params& p)
{
    return {
        real(p),
        imaginary(p)
    };
}

