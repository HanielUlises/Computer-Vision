// ShapeTracker.cpp
#include "ShapeTracker.hpp"

ShapeTracker::ShapeTracker() : lower_hue(cv::Scalar(29, 86, 6)), upper_hue(cv::Scalar(64, 255, 255)) {}

void ShapeTracker::trackCircle(cv::VideoCapture& capture) {
    cv::Mat frame;
    while (true) {
        capture >> frame;
        if (frame.empty()) break;
        processFrame(frame);
        if (cv::waitKey(10) >= 0) break;
    }
}

void ShapeTracker::processFrame(cv::Mat& frame) {
    cv::Mat blurred, hsv, mask;
    cv::GaussianBlur(frame, blurred, cv::Size(11, 11), 0);
    cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, lower_hue, upper_hue, mask);

    // Erode and dilate mask to remove noise
    cv::erode(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);
    cv::dilate(mask, mask, cv::Mat(), cv::Point(-1, -1), 2);

    // Find circles
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<float> radii;
    std::vector<cv::Point2f> centers;
    for (const auto& contour : contours) {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contour, center, radius);
        if (radius > 10) {
            cv::circle(frame, center, (int)radius, cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imshow("Detected Circles", frame);
}

