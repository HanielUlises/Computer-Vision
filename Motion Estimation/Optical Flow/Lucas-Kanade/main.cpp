#include <opencv2/opencv.hpp>
#include <iostream>

#include "pyramidal_lk.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./lk_tracker <video>" << std::endl;
        return -1;
    }

    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video." << std::endl;
        return -1;
    }

    cv::Mat prev_frame, curr_frame;
    cv::Mat prev_gray, curr_gray;

    cap >> prev_frame;
    if (prev_frame.empty()) return -1;

    cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> prev_pts;
    cv::goodFeaturesToTrack(
        prev_gray,
        prev_pts,
        500,
        0.01,
        10
    );

    PyramidalLK tracker;

    while (true) {
        cap >> curr_frame;
        if (curr_frame.empty()) break;

        cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> curr_pts;
        std::vector<uchar> status;

        tracker.track(prev_gray, curr_gray, prev_pts, curr_pts, status);

        for (size_t i = 0; i < curr_pts.size(); ++i) {
            if (status[i]) {
                cv::line(curr_frame, prev_pts[i], curr_pts[i],
                         cv::Scalar(0, 255, 0), 2);
                cv::circle(curr_frame, curr_pts[i],
                           3, cv::Scalar(0, 0, 255), -1);
            }
        }

        cv::imshow("Pyramidal LK Optical Flow", curr_frame);
        if (cv::waitKey(1) == 27) break;

        prev_gray = curr_gray.clone();
        prev_pts = curr_pts;
    }

    return 0;
}
