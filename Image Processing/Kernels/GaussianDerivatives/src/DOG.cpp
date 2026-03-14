#include "DOG.hpp"
#include <cmath>

static constexpr double PI=3.14159265358979323846;

double DOG::gaussian(double x,double y,double sigma){
    double s2=sigma*sigma;
    double norm=1.0/(2.0*PI*s2);
    double r2=x*x+y*y;

    return norm*std::exp(-r2/(2.0*s2));
}

Eigen::MatrixXd DOG::generate(
    int width,
    int height,
    double sigma1,
    double sigma2
){
    Eigen::MatrixXd kernel(height,width);

    int cx = width/2;
    int cy = height/2;

    for(int y = 0;y < height;++y){
        for(int x = 0;x < width;++x){

            double dx = x-cx;
            double dy = y-cy;

            double g1 = gaussian(dx,dy,sigma1);
            double g2 = gaussian(dx,dy,sigma2);

            kernel(y,x)=g1-g2;
        }
    }

    return kernel;
}