#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct optical_flow_params {
    int max_corners = 500;
    double quality = 0.01;
    double min_distance = 10.0;

    int win_size = 21;
    int max_level = 3;
    cv::TermCriteria term_criteria = 
        cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);

    double fb_threshold = 1.5;   // forward–backward check threshold
};

class optical_flow_tracker {
public:
    explicit optical_flow_tracker(const optical_flow_params& params = optical_flow_params());

    /// extract features in the first frame
    void initialize(const cv::Mat& frame_gray);

    /// process next frame — compute sparse + dense optical flow
    void track(const cv::Mat& next_frame_gray);

    /// get sparse points that survived FB consistency check
    const std::vector<cv::Point2f>& get_tracked_points() const;

    /// dense flow access
    const cv::Mat& get_dense_flow() const;

    /// draw visualization (arrows + tracks)
    cv::Mat visualize_flow(const cv::Mat& frame_bgr) const;

private:
    optical_flow_params params_;
    bool initialized_ = false;

    // sparse flow
    std::vector<cv::Point2f> prev_points_;
    std::vector<cv::Point2f> curr_points_;

    // dense flow (Farnebäck)
    cv::Mat dense_flow_;

    cv::Mat prev_frame_;
};

