#include "GaborComplex.hpp"
#include <cmath>

std::pair<cv::Mat, cv::Mat>
GaborComplex::create(const Params& p)
{
    CV_Assert(p.ksize % 2 == 1);
    int h = p.ksize / 2;

    cv::Mat real(p.ksize, p.ksize, CV_32F);
    cv::Mat imag(p.ksize, p.ksize, CV_32F);

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

        double phase = 2.0 * CV_PI * xr / p.lambda;

        real.at<float>(y+h, x+h) =
            static_cast<float>(gauss * std::cos(phase));

        imag.at<float>(y+h, x+h) =
            static_cast<float>(gauss * std::sin(phase));
    }

    // Zero-mean for R
    real -= cv::mean(real)[0];

    return { real, imag };
}
