#include <opencv2/opencv.hpp>
#include <iostream>

int main() 
{
    //Reading image
    std::string path = "/usercode/shapes1.jpg";
    cv::Mat img = cv::imread(path);
    cv::Mat src_gray;
    cv::cvtColor(img, src_gray, cv::COLOR_BGR2GRAY);

    //Canny ddge detection
    cv::Mat canny_img;
    cv::Canny(src_gray, canny_img, 50, 50);
    cv::imshow("Canny Image", canny_img);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(canny_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(img, contours, -1, cv::Scalar(255, 0, 255), 2);

    cv::imshow("Contoured Image", img);
    cv::waitKey(0);

    return 0;
}