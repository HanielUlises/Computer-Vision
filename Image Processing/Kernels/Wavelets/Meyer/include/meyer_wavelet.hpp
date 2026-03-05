#ifndef MEYER_WAVELET_HPP
#define MEYER_WAVELET_HPP

#include <vector>

class MeyerWavelet
{
public:
    // Generate 1D Meyer wavelet sampled in time domain
    static std::vector<double> generate(
        int size,
        double dt = 0.01
    );

private:
    static double nu(double x);
    static double psi_hat(double omega);
};

#endif