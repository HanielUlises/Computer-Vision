#include <opencv2/opencv.hpp>
#include "Morlet.hpp"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: morlet_demo <image_path>\n";
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cerr << "Failed to load image\n";
        return -1;
    }

    img.convertTo(img, CV_32F, 1.0 / 255.0);

    MorletWavelet::Params p;
    p.ksize     = 41;
    p.sigma     = 6.0;
    p.frequency = 0.15;
    p.theta     = CV_PI / 4.0;

    auto [kReal, kImag] = MorletWavelet::complex(p);

    cv::Mat respReal, respImag;
    cv::filter2D(img, respReal, CV_32F, kReal);
    cv::filter2D(img, respImag, CV_32F, kImag);

    cv::Mat magnitude;
    cv::magnitude(respReal, respImag, magnitude);

    cv::normalize(respReal, respReal, 0, 1, cv::NORM_MINMAX);
    cv::normalize(respImag, respImag, 0, 1, cv::NORM_MINMAX);
    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

    cv::imshow("Input", img);
    cv::imshow("Morlet Real Response", respReal);
    cv::imshow("Morlet Imag Response", respImag);
    cv::imshow("Morlet Magnitude", magnitude);

    cv::waitKey(0);
    return 0;
}
