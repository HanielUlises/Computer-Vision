#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <cmath>
#include <vector>

void draw_rect(int action, int x, int y, int flags, void *user_data) {
    static cv::Point init_point;  
    if (action == cv::EVENT_LBUTTONDOWN) {
        init_point = cv::Point(x, y);
    } else if (action == cv::EVENT_LBUTTONUP) {
        cv::Point end_point(x, y);
        cv::rectangle(*(cv::Mat*)user_data, init_point, end_point, cv::Scalar(255, 0, 0), 2, cv::LINE_8);

        cv::Rect roi(std::min(init_point.x, end_point.x), 
                     std::min(init_point.y, end_point.y), 
                     std::abs(end_point.x - init_point.x), 
                     std::abs(end_point.y - init_point.y));

        cv::Mat cropped = (*(cv::Mat*)user_data)(roi);
        cv::imshow("Window", *(cv::Mat*)user_data);
        cv::imwrite("output.jpg", cropped);
    }
}

int main() {
    cv::Mat source = cv::imread("sample.jpg", 1);
    if (source.empty()) {
        std::cout << "Error: Could not load image 'sample.jpg'" << std::endl;
        return -1;
    }

    cv::Mat dummy = source.clone();
    cv::namedWindow("Window");
    cv::setMouseCallback("Window", draw_rect, &source);
    int k = 0;
    while (k != 27) {
        imshow("Window", source);
        putText(source, "Choose center, and drag, Press ESC to exit and c to clear",
                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
        k = cv::waitKey(20) & 0xFF;
        if (k == 99) {
            dummy.copyTo(source);
        }
    }
    return 0;
}