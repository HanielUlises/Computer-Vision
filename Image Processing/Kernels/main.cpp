#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "kernel2d.hpp"
#include "kernels.hpp"
#include "opencv_bridge.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./kernel_test <image_path>\n";
        return 1;
    }

    cv::Mat img_u8 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img_u8.empty()) {
        std::cerr << "Failed to load image\n";
        return 1;
    }

    cv::Mat img;
    img_u8.convertTo(img, CV_32F, 1.0 / 255.0);
    cv::Mat out(img.size(), CV_32F, cv::Scalar(0));

    auto v = view<float>(img, out);

    convolve_ignore_border(
        v.src,
        v.dst,
        v.width,
        v.height,
        v.stride,
        SobelX );

    cv::Mat vis;
    cv::normalize(out, vis, 0, 1, cv::NORM_MINMAX);

    cv::imshow("Input", img);
    cv::imshow("SobelX (TMP)", vis);
    cv::waitKey(0);

    return 0;
}
