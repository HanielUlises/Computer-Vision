#pragma once
#include <opencv2/core.hpp>
#include <tuple>


class GaussianSecondDerivative
{
public:
    struct Params
    {
        int     ksize;
        double  sigma;
    };

    // ∂²G / ∂x²
    static cv::Mat dxx(const Params& p);

    // ∂²G / ∂y²
    static cv::Mat dyy(const Params& p);

    // ∂²G / ∂x∂y
    static cv::Mat dxy(const Params& p);
};

