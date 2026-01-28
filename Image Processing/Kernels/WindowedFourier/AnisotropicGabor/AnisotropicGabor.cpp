#include "AnisotropicGabor.hpp"
#include <cmath>

AnisotropicGabor::AnisotropicGabor(
    int kernelSize,
    double sigmaX,
    double sigmaY,
    double wavelength,
    double orientation,
    double phase
)
    : ksize(kernelSize),
      sx(sigmaX),
      sy(sigmaY),
      lambda(wavelength),
      theta(orientation),
      phi(phase)
{
    buildKernel();
}

void AnisotropicGabor::buildKernel()
{
    kernelMat = cv::Mat(ksize, ksize, CV_64F);
    int half = ksize / 2;

    double cosT = std::cos(theta);
    double sinT = std::sin(theta);

    for (int y = -half; y <= half; ++y)
    {
        for (int x = -half; x <= half; ++x)
        {
            double xr =  x * cosT + y * sinT;
            double yr = -x * sinT + y * cosT;

            double gauss =
                std::exp(
                    -(xr * xr) / (2.0 * sx * sx)
                    -(yr * yr) / (2.0 * sy * sy)
                );

            double carrier =
                std::cos(2.0 * M_PI * xr / lambda + phi);

            kernelMat.at<double>(y + half, x + half) =
                gauss * carrier;
        }
    }

    kernelMat -= cv::mean(kernelMat)[0];
}

void AnisotropicGabor::apply(
    const cv::Mat& src,
    cv::Mat& dst
) const
{
    cv::filter2D(src, dst, CV_64F, kernelMat);
}

const cv::Mat& AnisotropicGabor::kernel() const
{
    return kernelMat;
}
