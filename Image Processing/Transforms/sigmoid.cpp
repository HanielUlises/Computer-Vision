#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

cv::Mat sigmoidTransform(const cv::Mat &gray_img, float alpha, float delta) {
    cv::Mat new_image = gray_img.clone();

    for (int i = 0; i < gray_img.rows; i++) {
        for (int j = 0; j < gray_img.cols; j++) {
            new_image.at<float>(i, j) = 1.0f / (1.0f + std::exp(-alpha * (gray_img.at<float>(i, j) - delta)));
        }
    }
    return new_image;
}

int main (){
    std::string path = "kodim23.png";
    cv::Mat image = cv::imread(path);
    if(image.empty()) std::cout << "Image file not found" << std::endl;

    image.convertTo(image, CV_32F, 1.0/255);
    cv::Mat gray_img;
    cv::cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
    gray_img.convertTo(gray_img, CV_32F, 1.0 / 255.0);

    cv::Mat new_image = sigmoidTransform(gray_img, 10.0f, 0.6f);
    cv::normalize(new_image, new_image, 0, 1, cv::NORM_MINMAX);
    new_image.convertTo(new_image, CV_8U, 255.0);

    cv::imwrite("transformed_image.png",new_image);
}