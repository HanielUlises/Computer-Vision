#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct block_matching_params {
    int block_size = 16;
    int search_window = 32;        // search_window around each block
    int step = 8;                  // block grid step (for overlapping v sparse)
    enum class Metric { SAD, SSD, NCC } metric = Metric::SAD;
};

class BlockMatchingTracker {
public:
    explicit BlockMatchingTracker(const block_matching_params& p = block_matching_params());
    

    void initialize(const cv::Mat& frame_gray);
    void compute_motion(const cv::Mat& next_gray);

    const cv::Mat& get_motion_field() const { return motion_field_; }  // 2-channel: (dx, dy)

    cv::Mat visualize(const cv::Mat& frame_bgr, int stride = 4) const;

    block_matching_params params_;

private:
    
    cv::Mat prev_gray_;
    cv::Mat motion_field_;
};