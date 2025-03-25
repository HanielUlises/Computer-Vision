#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat frame_diff(const cv::Mat &prev_frame, const cv::Mat &cur_frame, const cv::Mat &next_frame)
{
    cv::Mat diff1, diff2, output;
    cv::absdiff(next_frame, cur_frame, diff1);
    cv::absdiff(cur_frame, prev_frame, diff2);
    cv::bitwise_and(diff1, diff2, output);
    return output;
}

cv::Mat get_frame(cv::VideoCapture &cap, float scaling_factor)
{
    cv::Mat frame, gray_frame;
    cap >> frame;
    cv::resize(frame, frame, cv::Size(), scaling_factor, scaling_factor, cv::INTER_AREA);
    cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
    return gray_frame;
}

int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    float scaling_factor = 0.75f;
    cv::Mat prev_frame = get_frame(cap, scaling_factor);
    cv::Mat cur_frame  = get_frame(cap, scaling_factor);
    cv::Mat next_frame = get_frame(cap, scaling_factor);

    cv::namedWindow("object_movement");

    while (true) {
        cv::imshow("object_movement", frame_diff(prev_frame, cur_frame, next_frame));
        prev_frame = cur_frame;
        cur_frame  = next_frame;
        next_frame = get_frame(cap, scaling_factor);
        if (cv::waitKey(30) == 27)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
