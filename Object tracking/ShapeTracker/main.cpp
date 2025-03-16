#include "ShapeTracker.h"
#include <iostream>

int main() {
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "ERROR: Could not open video capture" << std::endl;
        return 1;
    }

    ShapeTracker tracker;
    tracker.trackCircle(capture);
    return 0;
}
