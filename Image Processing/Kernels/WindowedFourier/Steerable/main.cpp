#include <opencv2/opencv.hpp>
#include "Steerable_Fourier.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./steerable_test <image_path>\n";
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cerr << "Could not load image\n";
        return -1;
    }

    img.convertTo(img, CV_64F, 1.0 / 255.0);

    SteerableFourier filter(
        31,     // kernel size
        6.0,    // sigma
        0.08    // frequency
    );

    std::vector<double> angles = {
        0.0,
        M_PI / 6.0,
        M_PI / 4.0,
        M_PI / 3.0,
        M_PI / 2.0
    };

    for (size_t i = 0; i < angles.size(); ++i)
    {
        cv::Mat response;
        filter.apply(img, response, angles[i]);

        cv::Mat vis;
        cv::normalize(response, vis, 0, 1, cv::NORM_MINMAX);

        std::string win =
            "Theta = " + std::to_string(angles[i]);

        cv::imshow(win, vis);
    }

    cv::waitKey(0);
    return 0;
}
