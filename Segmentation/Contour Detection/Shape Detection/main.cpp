#include <opencv2/opencv.hpp>
#include "contour_detector.hpp"
#include <iostream>

int main()
{
    std::string path = "/usercode/shapes1.jpg";
    cv::Mat img = cv::imread(path);

    if (img.empty())
    {
        std::cerr << "Error: Could not open or find the image!\n";
        return -1;
    }

    ContourDetector detector;
    detector.process(img);

    cv::Mat img_with_contours = img.clone();
    detector.draw_contours(img_with_contours);

    cv::imshow("Original Image", img);
    cv::imshow("Contours Detected", img_with_contours);
    cv::waitKey(0);

    return 0;
}
