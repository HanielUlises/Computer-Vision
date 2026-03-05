#include "MexicanHat.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>

MexicanHat::MexicanHat(int ksize, double sigma)
    : ksize_(ksize), sigma_(sigma)
{
    createKernel();
}

void MexicanHat::createKernel()
{
    kernel_ = cv::Mat(ksize_, ksize_, CV_32F);

    int r = ksize_ / 2;
    double sigma2 = sigma_ * sigma_;
    double sigma4 = sigma2 * sigma2;

    for (int y = -r; y <= r; ++y) {
        for (int x = -r; x <= r; ++x) {

            double r2 = x * x + y * y;

            // Mexican Hat (Laplacian of Gaussian form)
            double value =
                (r2 - 2.0 * sigma2) / sigma4 *
                std::exp(-r2 / (2.0 * sigma2));

            kernel_.at<float>(y + r, x + r) =
                static_cast<float>(value);
        }
    }

    cv::Scalar sum = cv::sum(kernel_);
    if (std::abs(sum[0]) > 1e-6)
        kernel_ -= sum[0] / (ksize_ * ksize_);
}

void MexicanHat::apply(const cv::Mat& src, cv::Mat& dst) const
{
    CV_Assert(src.type() == CV_32F);

    cv::filter2D(src, dst, CV_32F, kernel_);
}