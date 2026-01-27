#include "GaborOdd.hpp"
#include <cmath>

cv::Mat GaborOdd::create(const Params& p)
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
        double xr =  x * ct + y * st;
        double yr = -x * st + y * ct;

        double gauss = std::exp(
            -(xr*xr + p.gamma*p.gamma*yr*yr) /
            (2.0 * sigma2)
        );

        double wave = std::sin(2.0 * CV_PI * xr / p.lambda);

        k.at<float>(y+h, x+h) =
            static_cast<float>(gauss * wave);
    }

    return k;
}
