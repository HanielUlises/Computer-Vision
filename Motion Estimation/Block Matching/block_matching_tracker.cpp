#include "block_matching_tracker.hpp"
#include <iostream>

BlockMatchingTracker::BlockMatchingTracker(const block_matching_params& p) : params_(p) {}

void BlockMatchingTracker::initialize(const cv::Mat& frame_gray) {
    prev_gray_ = frame_gray.clone();
    motion_field_ = cv::Mat::zeros(frame_gray.size(), CV_32FC2);
}

void BlockMatchingTracker::compute_motion(const cv::Mat& next_gray) {
    if (prev_gray_.empty()) {
        initialize(next_gray);
        return;
    }

    motion_field_.setTo(0);

    int bs = params_.block_size;
    int hw = bs / 2;
    int sw = params_.search_window;

    for (int y = hw; y < prev_gray_.rows - hw; y += params_.step) {
        for (int x = hw; x < prev_gray_.cols - hw; x += params_.step) {
            cv::Rect block_rect(x - hw, y - hw, bs, bs);
            cv::Mat block = prev_gray_(block_rect);

            int best_x = x, best_y = y;
            float best_score = std::numeric_limits<float>::max();

            for (int dy = -sw; dy <= sw; ++dy) {
                for (int dx = -sw; dx <= sw; ++dx) {
                    int nx = x + dx, ny = y + dy;
                    if (nx - hw < 0 || nx + hw >= next_gray.cols ||
                        ny - hw < 0 || ny + hw >= next_gray.rows) continue;

                    cv::Rect candidate(nx - hw, ny - hw, bs, bs);
                    cv::Mat cand = next_gray(candidate);

                    float score = 0;
                    if (params_.metric == block_matching_params::Metric::SAD) {
                        score = cv::norm(block, cand, cv::NORM_L1);
                    } else if (params_.metric == block_matching_params::Metric::SSD) {
                        score = cv::norm(block, cand, cv::NORM_L2SQR);
                    }

                    if (score < best_score) {
                        best_score = score;
                        best_x = nx;
                        best_y = ny;
                    }
                }
            }

            motion_field_.at<cv::Point2f>(y, x) = cv::Point2f(best_x - x, best_y - y);
        }
    }

    prev_gray_ = next_gray.clone();
    std::cout << "[BlockMatching] Computed motion field (non-overlapping blocks)\n";
}

cv::Mat BlockMatchingTracker::visualize(const cv::Mat& frame_bgr, int stride) const {
    cv::Mat vis = frame_bgr.clone();

    for (int y = 0; y < motion_field_.rows; y += stride) {
        for (int x = 0; x < motion_field_.cols; x += stride) {
            cv::Point2f m = motion_field_.at<cv::Point2f>(y, x);
            if (cv::norm(m) < 0.5) continue;

            cv::Point p1(x, y);
            cv::Point p2(cvRound(x + m.x), cvRound(y + m.y));
            cv::arrowedLine(vis, p1, p2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.3);
            cv::circle(vis, p1, 2, cv::Scalar(255, 0, 0), -1);
        }
    }
    return vis;
}