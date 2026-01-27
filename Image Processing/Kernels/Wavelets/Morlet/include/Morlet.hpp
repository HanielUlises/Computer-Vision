#pragma once

#include <opencv2/core.hpp>
#include <utility>

/**
 * @brief 2D Morlet wavelet (directional, admissible).
 *
 * Produces real and imaginary kernels suitable for
 * convolution-based wavelet analysis.
 */
class MorletWavelet
{
public:
    struct Params
    {
        int     ksize;      // Kernel size (odd)
        double  sigma;      // Gaussian scale
        double  frequency;  // Central frequency (cycles / pixel)
        double  theta;      // Orientation (radians)
    };

    static cv::Mat real(const Params& p);
    static cv::Mat imaginary(const Params& p);
    static std::pair<cv::Mat, cv::Mat> complex(const Params& p);
};
