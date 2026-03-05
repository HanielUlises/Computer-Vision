#ifndef MEYER2D_HPP
#define MEYER2D_HPP

#include <Eigen/Dense>
#include <complex>

class Meyer2D
{
public:
    using Complex = std::complex<double>;
    using MatrixC = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

    static MatrixC generate(
        int width,
        int height,
        double scale
    );

private:
    static double nu(double x);
    static double psi_hat_radial(double omega);
};

#endif