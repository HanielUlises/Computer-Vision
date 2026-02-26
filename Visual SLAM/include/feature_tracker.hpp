#pragma once

#include "slam_types.hpp"
#include <opencv2/features2d.hpp>

namespace vslam {

class FeatureTracker {
public:
    explicit FeatureTracker(const SLAMConfig& cfg);

    void detect_and_compute(Frame::Ptr& frame);

    std::vector<cv::DMatch> match(const Frame::Ptr& ref,
                                  const Frame::Ptr& cur) const;

    std::vector<cv::DMatch> match_with_optical_flow(const Frame::Ptr& ref,
                                                     const Frame::Ptr& cur,
                                                     std::vector<uchar>& status) const;

    std::vector<cv::DMatch> filter_matches_ratio(
        const std::vector<std::vector<cv::DMatch>>& knn_matches) const;

private:
    cv::Ptr<cv::ORB>          detector_;
    cv::Ptr<cv::BFMatcher>    matcher_;
    SLAMConfig                cfg_;

    std::vector<cv::DMatch> filter_by_homography(
        const std::vector<cv::KeyPoint>& kp1,
        const std::vector<cv::KeyPoint>& kp2,
        const std::vector<cv::DMatch>& matches) const;
};

} // namespace vslam