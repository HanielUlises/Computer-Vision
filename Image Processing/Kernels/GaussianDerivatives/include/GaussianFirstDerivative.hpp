#pragma once
#include <opencv2/core.hpp>

class GaussianFirstDerivative
{
public:
    struct Params
    {
        int     ksize;   // Odd kernel size
        double  sigma;   // Scale
        double  theta;   // Orientation (radians)
    };

    /// ∂G/∂xθ (directional derivative)
    static cv::Mat create(const Params& p);
};