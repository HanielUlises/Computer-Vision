#ifndef DOG_HPP
#define DOG_HPP

#include <Eigen/Dense>

class DOG{
    public:

        static Eigen::MatrixXd generate(
            int width,
            int height,
            double sigma1,
            double sigma2
        );

    private:

        static double gaussian(
            double x,
            double y,
            double sigma
        );
};

#endif