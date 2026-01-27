#pragma once
#include <opencv2/core.hpp>

class GaborOdd
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

    static cv::Mat create(const Params& p);
};
