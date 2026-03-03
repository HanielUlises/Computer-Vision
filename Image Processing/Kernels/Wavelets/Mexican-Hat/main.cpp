#include <iostream>
#include <opencv2/opencv.hpp>
#include "MexicanHat.hpp"

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "Usage: ./MexicanHatDemo <image_path>\n";
        return 1;
    }

    cv::Mat img_u8 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img_u8.empty()) {
        std::cout << "Failed to load image.\n";
        return 1;
    }

    cv::Mat img;
    img_u8.convertTo(img, CV_32F, 1.0 / 255.0);

    MexicanHat mh(31, 4.0);   // kernel size, SIGMA
    cv::Mat output;

    mh.apply(img, output);

    cv::Mat vis;
    cv::normalize(output, vis, 0, 1, cv::NORM_MINMAX);

    cv::imshow("Input", img);
    cv::imshow("Mexican Hat", vis);
    cv::waitKey(0);

    return 0;
}