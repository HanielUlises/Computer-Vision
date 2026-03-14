#include <opencv2/opencv.hpp>
#include <iostream>
#include <Eigen/Dense>
#include "GaussianDerivatives/include/DOG.hpp"

int main(){

    int ksize=31;

    double sigma1=1.0;
    double sigma2=2.0;

    Eigen::MatrixXd kernel=DOG::generate(
        ksize,
        ksize,
        sigma1,
        sigma2
    );

    cv::Mat kernelMat(
        ksize,
        ksize,
        CV_64F,
        kernel.data()
    );

    cv::Mat image=cv::imread(
        "image.png",
        cv::IMREAD_GRAYSCALE
    );

    if(image.empty()){
        std::cout<<"image not found\n";
        return 0;
    }

    cv::Mat result;

    cv::filter2D(
        image,
        result,
        CV_64F,
        kernelMat
    );

    cv::imshow("input",image);
    cv::imshow("DOG result",result);

    cv::waitKey(0);

    return 0;
}