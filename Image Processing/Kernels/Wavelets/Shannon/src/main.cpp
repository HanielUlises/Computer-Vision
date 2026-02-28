#include "shannon.hpp"

#include <opencv2/imgproc.hpp>
#include <cmath>
#include <vector>

Shannon::Shannon(double f_low, double f_high)
    : f_low_(f_low), f_high_(f_high)
{
}

void Shannon::fftShift(cv::Mat& img)
{
    int cx = img.cols / 2;
    int cy = img.rows / 2;

    cv::Mat q0(img, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(img, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(img, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(img, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp); q3.copyTo(q0); tmp.copyTo(q3);
    q1.copyTo(tmp); q2.copyTo(q1); tmp.copyTo(q2);
}

void Shannon::createBandPassMask(cv::Mat& mask) const
{
    int rows = mask.rows;
    int cols = mask.cols;

    double cx = cols / 2.0;
    double cy = rows / 2.0;
    double maxRadius = std::sqrt(cx * cx + cy * cy);

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {

            double dx = x - cx;
            double dy = y - cy;
            double r = std::sqrt(dx * dx + dy * dy);

            double normFreq = r / maxRadius;

            mask.at<float>(y, x) =
                (normFreq >= f_low_ && normFreq <= f_high_) ? 1.0f : 0.0f;
        }
    }
}

void Shannon::apply(const cv::Mat& src, cv::Mat& dst) const
{
    CV_Assert(src.type() == CV_32F);

    cv::Mat complexImg;
    cv::dft(src, complexImg, cv::DFT_COMPLEX_OUTPUT);

    fftShift(complexImg);

    cv::Mat mask(src.size(), CV_32F);
    createBandPassMask(mask);

    std::vector<cv::Mat> channels(2);
    cv::split(complexImg, channels);

    channels[0] = channels[0].mul(mask);
    channels[1] = channels[1].mul(mask);

    cv::merge(channels, complexImg);

    fftShift(complexImg);

    cv::idft(complexImg, dst,
             cv::DFT_REAL_OUTPUT | cv::DFT_SCALE);
}