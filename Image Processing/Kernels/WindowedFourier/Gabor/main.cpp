#include <opencv2/opencv.hpp>
#include "GaborEven.hpp"
#include "GaborOdd.hpp"
#include "GaborComplex.hpp"

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: gabor_demo <image_path>\n";
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cerr << "Could not load image\n";
        return -1;
    }

    img.convertTo(img, CV_32F, 1.0 / 255.0);

    int ksize      = 41;
    double sigma   = 6.0;
    double theta   = CV_PI / 4.0;   // 45°
    double lambda  = 12.0;          // wavelength in pixels
    double gamma   = 0.5;           // ellipticity

    GaborEven::Params pEven{ ksize, sigma, theta, lambda, gamma };
    cv::Mat kEven = GaborEven::create(pEven);

    cv::Mat respEven;
    cv::filter2D(img, respEven, CV_32F, kEven);

    GaborOdd::Params pOdd{ ksize, sigma, theta, lambda, gamma };
    cv::Mat kOdd = GaborOdd::create(pOdd);

    cv::Mat respOdd;
    cv::filter2D(img, respOdd, CV_32F, kOdd);

    GaborComplex::Params pC{ ksize, sigma, theta, lambda, gamma };
    auto [kReal, kImag] = GaborComplex::create(pC);

    cv::Mat respReal, respImag;
    cv::filter2D(img, respReal, CV_32F, kReal);
    cv::filter2D(img, respImag, CV_32F, kImag);

    cv::Mat magnitude;
    cv::magnitude(respReal, respImag, magnitude);

    cv::normalize(respEven, respEven, 0, 1, cv::NORM_MINMAX);
    cv::normalize(respOdd, respOdd, 0, 1, cv::NORM_MINMAX);
    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);

    cv::imshow("Input", img);
    cv::imshow("Gabor Even Response", respEven);
    cv::imshow("Gabor Odd Response", respOdd);
    cv::imshow("Gabor Magnitude (Complex)", magnitude);

    cv::waitKey(0);
    return 0;
}
