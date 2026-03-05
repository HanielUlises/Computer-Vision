#include "meyer_wavelet.hpp"
#include <cmath>

static constexpr double PI = 3.14159265358979323846;

double MeyerWavelet::nu(double x)
{
    // Smooth transition polynomial
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;

    return x * x * x * (10 - 15 * x + 6 * x * x);
}

double MeyerWavelet::psi_hat(double omega)
{
    double abs_omega = std::abs(omega);

    if (abs_omega <= 2.0 * PI / 3.0)
        return 0.0;

    if (abs_omega >= 4.0 * PI / 3.0)
        return 0.0;

    if (abs_omega <= PI)
    {
        double t = (3.0 * abs_omega / (2.0 * PI)) - 1.0;
        return std::sin(PI / 2.0 * nu(t));
    }

    if (abs_omega <= 4.0 * PI / 3.0)
    {
        double t = (3.0 * abs_omega / (2.0 * PI)) - 1.0;
        return std::cos(PI / 2.0 * nu(t));
    }

    return 0.0;
}

std::vector<double> MeyerWavelet::generate(int size, double dt)
{
    std::vector<double> wavelet(size);

    int half = size / 2;
    double domega = 2.0 * PI / (size * dt);

    for (int n = 0; n < size; ++n)
    {
        double t = (n - half) * dt;
        double sum = 0.0;

        for (int k = -half; k < half; ++k)
        {
            double omega = k * domega;
            double value = psi_hat(omega);

            sum += value * std::cos(omega * t);
        }

        wavelet[n] = sum * domega / (2.0 * PI);
    }

    return wavelet;
}