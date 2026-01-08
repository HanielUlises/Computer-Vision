#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class PyramidalLK {
    public:
        PyramidalLK(
            int win_size = 21,
            int max_level = 3,
            int max_iters = 30,
            double epsilon = 0.01
        );

        void track(
            const cv::Mat& prev_img,
            const cv::Mat& curr_img,
            std::vector<cv::Point2f>& prev_pts,
            std::vector<cv::Point2f>& curr_pts,
            std::vector<uchar>& status
        );

    private:
        cv::Size win_size_;
        int max_level_;
        cv::TermCriteria termcrit_;
};
