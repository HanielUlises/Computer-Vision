#include "farneback_tracker.hpp"
#include <iostream>

FarnebackTracker::FarnebackTracker(const farneback_params& params) : params_(params) {}

void FarnebackTracker::initialize(const cv::Mat& frame_gray) {
    cv::goodFeaturesToTrack(frame_gray, prev_points_,
                            params_.max_corners, params_.quality_level, params_.min_distance);

    frame_gray.copyTo(prev_gray_);
    curr_points_ = prev_points_;
    initialized_ = true;

    std::cout << "[FarnebackTracker] Initialized with " << prev_points_.size() << " corners.\n";
}

void FarnebackTracker::track(const cv::Mat& next_frame_gray) {
    if (!initialized_ || prev_points_.empty()) {
        initialize(next_frame_gray);
        return;
    }

    std::vector<cv::Point2f> fwd_points;
    std::vector<uchar> status_fwd, status_bwd;
    std::vector<float> err;

    // Forward LK
    cv::calcOpticalFlowPyrLK(prev_gray_, next_frame_gray, prev_points_, fwd_points,
                             status_fwd, err, cv::Size(params_.win_size, params_.win_size),
                             params_.max_level, params_.term_criteria);

    // Backward LK for consistency
    std::vector<cv::Point2f> bwd_points;
    cv::calcOpticalFlowPyrLK(next_frame_gray, prev_gray_, fwd_points, bwd_points,
                             status_bwd, err);

    // FB error filtering
    curr_points_.clear();
    for (size_t i = 0; i < fwd_points.size(); ++i) {
        if (status_fwd[i] && status_bwd[i]) {
            double fb_err = cv::norm(prev_points_[i] - bwd_points[i]);
            if (fb_err < params_.fb_threshold) {
                curr_points_.push_back(fwd_points[i]);
            }
        }
    }

    // Dense Farnebäck flow
    cv::calcOpticalFlowFarneback(prev_gray_, next_frame_gray, dense_flow_,
                                 params_.pyr_scale, params_.levels, params_.winsize,
                                 params_.iterations, params_.poly_n, params_.poly_sigma, params_.flags);

    prev_points_ = curr_points_;
    prev_gray_ = next_frame_gray.clone();

    std::cout << "[FarnebackTracker] Tracked " << curr_points_.size() << " reliable points.\n";
}

cv::Mat FarnebackTracker::visualize(const cv::Mat& frame_bgr) const {
    cv::Mat vis = frame_bgr.clone();

    // Draw sparse tracks
    for (size_t i = 0; i < std::min(prev_points_.size(), curr_points_.size()); ++i) {
        cv::arrowedLine(vis, prev_points_[i], curr_points_[i],
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA, 0, 0.3);
        cv::circle(vis, curr_points_[i], 3, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
    }

    // Draw dense flow (HSV)
    if (!dense_flow_.empty()) {
        cv::Mat flow_hsv(dense_flow_.size(), CV_8UC3);
        for (int y = 0; y < dense_flow_.rows; ++y) {
            for (int x = 0; x < dense_flow_.cols; ++x) {
                cv::Point2f f = dense_flow_.at<cv::Point2f>(y, x);
                float mag = std::hypot(f.x, f.y);
                float ang = std::atan2(f.y, f.x) * 180 / CV_PI + 180; // 0–360

                flow_hsv.at<cv::Vec3b>(y, x) = {
                    static_cast<uchar>(ang / 2),     // H
                    255,                             // S
                    static_cast<uchar>(std::min(255.0f, mag * 8.0f))  // V
                };
            }
        }
        cv::Mat flow_bgr;
        cv::cvtColor(flow_hsv, flow_bgr, cv::COLOR_HSV2BGR);
        cv::addWeighted(vis, 0.8, flow_bgr, 0.2, 0.0, vis);
    }

    return vis;
}