#pragma once
#include <opencv2/core.hpp>
#include <utility>

class GaborComplex
{
public:
    struct Params
    {
        int     ksize;
        double  sigma;
        double  theta;
        double  lambda;
        double  gamma;
    };

    static std::pair<cv::Mat, cv::Mat> create(const Params& p);
};
