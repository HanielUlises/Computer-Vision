#include "optical_flow.hpp"

optical_flow_tracker::optical_flow_tracker(const optical_flow_params& params)
    : params_(params) {}

void optical_flow_tracker::initialize(const cv::Mat& frame_gray) {
    cv::goodFeaturesToTrack(
        frame_gray,
        prev_points_,
        params_.max_corners,
        params_.quality,
        params_.min_distance
    );

    frame_gray.copyTo(prev_frame_);
    initialized_ = true;
}

void optical_flow_tracker::track(const cv::Mat& next_frame_gray) {
    if (!initialized_) {
        initialize(next_frame_gray);
        return;
    }

    std::vector<cv::Point2f> fwd_points, bwd_points;
    std::vector<unsigned char> status_fwd, status_bwd;
    std::vector<float> err_fwd, err_bwd;

    // FORWARD optical flow
    cv::calcOpticalFlowPyrLK(
        prev_frame_, next_frame_gray,
        prev_points_, fwd_points,
        status_fwd, err_fwd,
        cv::Size(params_.win_size, params_.win_size),
        params_.max_level,
        params_.term_criteria
    );

    // BACKWARD optical flow (for outlier rejection)
    cv::calcOpticalFlowPyrLK(
        next_frame_gray, prev_frame_,
        fwd_points, bwd_points,
        status_bwd, err_bwd,
        cv::Size(params_.win_size, params_.win_size),
        params_.max_level,
        params_.term_criteria
    );

    // Forward–Backward consistency check
    curr_points_.clear();
    for (size_t i = 0; i < prev_points_.size(); ++i) {
        if (!status_fwd[i] || !status_bwd[i]) continue;

        double fb_error = cv::norm(prev_points_[i] - bwd_points[i]);
        if (fb_error < params_.fb_threshold) {
            curr_points_.push_back(fwd_points[i]);
        }
    }

    // Dense Flow (Farnebäck) 
    cv::calcOpticalFlowFarneback(
        prev_frame_, next_frame_gray,
        dense_flow_,
        0.5,           // pyramid scale
        3,             // number of levels
        15,            // window size
        3,             // iterations
        5,             // poly_n
        1.2,           // poly_sigma
        0              // flags
    );

    prev_points_ = curr_points_;
    next_frame_gray.copyTo(prev_frame_);
}

const std::vector<cv::Point2f>& optical_flow_tracker::get_tracked_points() const {
    return curr_points_;
}

const cv::Mat& optical_flow_tracker::get_dense_flow() const {
    return dense_flow_;
}

cv::Mat optical_flow_tracker::visualize_flow(const cv::Mat& frame_bgr) const {
    cv::Mat vis = frame_bgr.clone();

    // Sparse flow (arrows)
    for (size_t i = 0; i < prev_points_.size() && i < curr_points_.size(); ++i) {
        cv::arrowedLine(
            vis,
            prev_points_[i],
            curr_points_[i],
            cv::Scalar(0, 255, 0),
            2
        );
    }

    // Dense flow visualization (HSV)
    cv::Mat hsv(vis.size(), CV_8UC3);
    for (int y = 0; y < dense_flow_.rows; ++y) {
        for (int x = 0; x < dense_flow_.cols; ++x) {
            const cv::Point2f& f = dense_flow_.at<cv::Point2f>(y, x);
            float magnitude = std::sqrt(f.x * f.x + f.y * f.y);
            float angle = std::atan2(f.y, f.x); // -pi to pi

            hsv.at<cv::Vec3b>(y, x)[0] = static_cast<unsigned char>((angle + CV_PI) * 90 / CV_PI);
            hsv.at<cv::Vec3b>(y, x)[1] = 255;
            hsv.at<cv::Vec3b>(y, x)[2] = static_cast<unsigned char>(std::min(255.0f, magnitude * 10));
        }
    }

    cv::Mat dense_bgr;
    cv::cvtColor(hsv, dense_bgr, cv::COLOR_HSV2BGR);

    cv::addWeighted(vis, 0.7, dense_bgr, 0.3, 0, vis);
    return vis;
}