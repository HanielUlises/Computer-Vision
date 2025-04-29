#include "contour_detector.hpp"
#include <cmath>

ContourDetector::ContourDetector()
{
}

void ContourDetector::process(const cv::Mat& input_image)
{
    convert_to_grayscale(input_image);
    apply_gaussian_blur();
    apply_manual_sobel();
    apply_threshold();
    find_contours();
}

void ContourDetector::draw_contours(cv::Mat& output_image) const
{
    cv::drawContours(output_image, contours_, -1, cv::Scalar(255, 0, 255), 2);
}

void ContourDetector::convert_to_grayscale(const cv::Mat& input_image)
{
    gray_image_ = cv::Mat(input_image.rows, input_image.cols, CV_8UC1);
    for (int i = 0; i < input_image.rows; ++i)
    {
        for (int j = 0; j < input_image.cols; ++j)
        {
            cv::Vec3b intensity = input_image.at<cv::Vec3b>(i, j);
            uchar gray = static_cast<uchar>(0.299 * intensity[2] + 0.587 * intensity[1] + 0.114 * intensity[0]);
            gray_image_.at<uchar>(i, j) = gray;
        }
    }
}

void ContourDetector::apply_gaussian_blur()
{
    int kernel_size = 5;
    double sigma = 1.0;
    int k = kernel_size / 2;
    cv::Mat kernel(kernel_size, kernel_size, CV_64F);

    // Create Gaussian Kernel
    double sum = 0.0;
    for (int x = -k; x <= k; ++x)
    {
        for (int y = -k; y <= k; ++y)
        {
            double value = (1.0 / (2 * M_PI * sigma * sigma)) * exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel.at<double>(x + k, y + k) = value;
            sum += value;
        }
    }

    kernel /= sum; // Normalize kernel

    blurred_image_ = cv::Mat(gray_image_.rows, gray_image_.cols, CV_8UC1);

    for (int i = k; i < gray_image_.rows - k; ++i)
    {
        for (int j = k; j < gray_image_.cols - k; ++j)
        {
            double sum = 0.0;
            for (int u = -k; u <= k; ++u)
            {
                for (int v = -k; v <= k; ++v)
                {
                    sum += gray_image_.at<uchar>(i + u, j + v) * kernel.at<double>(u + k, v + k);
                }
            }
            blurred_image_.at<uchar>(i, j) = static_cast<uchar>(sum);
        }
    }
}

void ContourDetector::apply_manual_sobel()
{
    edges_ = cv::Mat(blurred_image_.rows, blurred_image_.cols, CV_8UC1, cv::Scalar(0));
    int gx_kernel[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int gy_kernel[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };

    for (int i = 1; i < blurred_image_.rows - 1; ++i)
    {
        for (int j = 1; j < blurred_image_.cols - 1; ++j)
        {
            int gx = 0, gy = 0;

            for (int u = -1; u <= 1; ++u)
            {
                for (int v = -1; v <= 1; ++v)
                {
                    gx += blurred_image_.at<uchar>(i + u, j + v) * gx_kernel[u + 1][v + 1];
                    gy += blurred_image_.at<uchar>(i + u, j + v) * gy_kernel[u + 1][v + 1];
                }
            }

            int magnitude = static_cast<int>(sqrt(gx * gx + gy * gy));
            magnitude = std::min(255, magnitude);
            edges_.at<uchar>(i, j) = static_cast<uchar>(magnitude);
        }
    }
}

void ContourDetector::apply_threshold()
{
    uchar threshold_value = 100;
    for (int i = 0; i < edges_.rows; ++i)
    {
        for (int j = 0; j < edges_.cols; ++j)
        {
            edges_.at<uchar>(i, j) = (edges_.at<uchar>(i, j) > threshold_value) ? 255 : 0;
        }
    }
}

void ContourDetector::find_contours()
{
    contours_.clear();
    cv::findContours(edges_, contours_, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
}
