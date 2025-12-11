#include "block_matching_tracker.hpp"
#include <opencv2/highgui.hpp>

int main(int argc, char** argv) {
    cv::VideoCapture cap(argc > 1 ? argv[1] : 0);
    if (!cap.isOpened()) return -1;

    BlockMatchingTracker bm;
    bm.params_.block_size = 16;
    bm.params_.search_window = 24;
    bm.params_.step = 12;

    cv::Mat frame, gray;
    while (cap.read(frame)) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        if (bm.get_motion_field().empty()) {
            bm.initialize(gray);
        } else {
            bm.compute_motion(gray);
        }

        cv::Mat vis = bm.visualize(frame, 12);
        cv::putText(vis, "Block Matching Motion Estimation", cv::Point(20, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0,255,255), 2);

        cv::imshow("Block Matching (True Non-Optical-Flow)", vis);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}