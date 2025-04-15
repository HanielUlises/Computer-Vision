// ShapeTracker.h
#ifndef SHAPE_TRACKER_H
#define SHAPE_TRACKER_H

#include <opencv2/opencv.hpp>

class ShapeTracker {
public:
    ShapeTracker();
    void trackCircle(cv::VideoCapture& capture);

private:
    cv::Scalar lower_hue;  // Lower color range for circle detection
    cv::Scalar upper_hue;  // Upper color range for circle detection
    void processFrame(cv::Mat& frame);
};

#endif // SHAPE_TRACKER_H
