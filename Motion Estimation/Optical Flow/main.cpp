#include <opencv2/opencv.hpp>
#include "optical_flow.hpp"

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) return -1;

    optical_flow_params params;
    optical_flow_tracker tracker(params);

    cv::Mat frame, gray;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        tracker.track(gray);

        cv::Mat vis = tracker.visualize_flow(frame);
        cv::imshow("Optical Flow", vis);

        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}
