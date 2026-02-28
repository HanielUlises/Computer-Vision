#include <iostream>
#include <opencv2/opencv.hpp>
#include "shannon.hpp"

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: ./ShannonDemo <image_path>\n";
        return 1;
    }

    cv::Mat img_u8 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img_u8.empty()) {
        std::cout << "Failed to load image.\n";
        return 1;
    }

    cv::Mat img;
    img_u8.convertTo(img, CV_32F, 1.0 / 255.0);

    Shannon sh(0.1, 0.3);   // band-pass range
    cv::Mat output;

    sh.apply(img, output);

    cv::Mat vis;
    cv::normalize(output, vis, 0, 1, cv::NORM_MINMAX);
    cv::imshow("Input", img);
    cv::imshow("Shannon Output", vis);
    cv::waitKey(0);

    return 0;
}