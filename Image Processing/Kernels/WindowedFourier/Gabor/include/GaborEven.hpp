#pragma once
#include <opencv2/core.hpp>

class GaborEven
{
public:
    struct Params
    {
        int     ksize;   // odd
        double  sigma;   // Gaussian standard
        double  theta;   // radians
        double  lambda;  // wavelength (pixels)
        double  gamma;   // aspect ratio
    };

    static cv::Mat create(const Params& p);
};
