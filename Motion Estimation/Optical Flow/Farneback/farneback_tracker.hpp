#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct farneback_params {
    // Sparse KLT
    int max_corners = 500;
    double quality_level = 0.01;
    double min_distance = 10.0;
    int win_size = 21;
    int max_level = 3;
    cv::TermCriteria term_criteria =
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    
    // pixels
    double fb_threshold = 1.5; 

    // Farnebäck dense flow
    double pyr_scale = 0.5;
    int levels = 3;
    int winsize = 15;
    int iterations = 3;
    int poly_n = 5;
    double poly_sigma = 1.2;
    int flags = 0;
};

class FarnebackTracker {
    public:
        explicit FarnebackTracker(const farneback_params& params = farneback_params());

        void initialize(const cv::Mat& frame_gray);
        void track(const cv::Mat& next_frame_gray);

        const std::vector<cv::Point2f>& get_tracked_points() const { return curr_points_; }
        const cv::Mat& get_dense_flow() const { return dense_flow_; }

        cv::Mat visualize(const cv::Mat& frame_bgr) const;

    private:
        farneback_params params_;
        bool initialized_ = false;

        cv::Mat prev_gray_;
        std::vector<cv::Point2f> prev_points_;
        std::vector<cv::Point2f> curr_points_;
        cv::Mat dense_flow_;
};