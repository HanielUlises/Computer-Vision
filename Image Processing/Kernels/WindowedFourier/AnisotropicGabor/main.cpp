#include <opencv2/opencv.hpp>
#include "AnisotropicGabor.hpp"
#include <iostream>
#include <vector>
#include <cmath>

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./anisotropic_gabor_test <image_path>\n";
        return -1;
    }

    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cerr << "Could not load image\n";
        return -1;
    }

    img.convertTo(img, CV_64F, 1.0 / 255.0);

    int ksize = 31;
    double sigmaX = 8.0;   // elongation direction
    double sigmaY = 3.0;   // orthogonal direction
    double wavelength = 10.0;

    std::vector<double> orientations = {
        0.0,
        M_PI / 6.0,
        M_PI / 4.0,
        M_PI / 3.0,
        M_PI / 2.0
    };

    for (size_t i = 0; i < orientations.size(); ++i)
    {
        AnisotropicGabor gabor(
            ksize,
            sigmaX,
            sigmaY,
            wavelength,
            orientations[i]
        );

        cv::Mat response;
        gabor.apply(img, response);

        cv::Mat vis;
        cv::normalize(response, vis, 0, 1, cv::NORM_MINMAX);

        std::string win =
            "Theta = " + std::to_string(orientations[i]);

        cv::imshow(win, vis);
    }

    cv::waitKey(0);
    return 0;
}
