#include "feature_tracker.hpp"
#include <opencv2/video/tracking.hpp>

namespace vslam {

FeatureTracker::FeatureTracker(const SLAMConfig& cfg) : cfg_(cfg) {
    detector_ = cv::ORB::create(
        cfg.num_features,
        cfg.scale_factor,
        cfg.num_pyramid_levels,
        31, 0, 2,
        cv::ORB::HARRIS_SCORE,
        31,
        cfg.fast_threshold
    );
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, false);
}

void FeatureTracker::detect_and_compute(Frame::Ptr& frame) {
    cv::Mat gray;
    if (frame->image.channels() == 3)
        cv::cvtColor(frame->image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = frame->image;

    detector_->detectAndCompute(gray, cv::noArray(),
                                 frame->keypoints, frame->descriptors);
}

std::vector<cv::DMatch> FeatureTracker::match(const Frame::Ptr& ref,
                                               const Frame::Ptr& cur) const {
    std::vector<std::vector<cv::DMatch>> knn;
    matcher_->knnMatch(ref->descriptors, cur->descriptors, knn, 2);
    return filter_matches_ratio(knn);
}

std::vector<cv::DMatch> FeatureTracker::match_with_optical_flow(
    const Frame::Ptr& ref,
    const Frame::Ptr& cur,
    std::vector<uchar>& status) const
{
    auto to_gray = [](const cv::Mat& img) {
        cv::Mat g;
        if (img.channels() == 3) cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
        else g = img;
        return g;
    };

    std::vector<cv::Point2f> pts_ref, pts_cur;
    cv::KeyPoint::convert(ref->keypoints, pts_ref);

    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
        to_gray(ref->image), to_gray(cur->image),
        pts_ref, pts_cur, status, err,
        cv::Size(21, 21), 3,
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01)
    );

    std::vector<cv::DMatch> matches;
    for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i]) continue;
        cur->keypoints.emplace_back(pts_cur[i], ref->keypoints[i].size,
                                    ref->keypoints[i].angle,
                                    ref->keypoints[i].response,
                                    ref->keypoints[i].octave);
        matches.push_back({static_cast<int>(i),
                           static_cast<int>(cur->keypoints.size() - 1),
                           0.0f});
    }
    return matches;
}

std::vector<cv::DMatch> FeatureTracker::filter_matches_ratio(
    const std::vector<std::vector<cv::DMatch>>& knn_matches) const
{
    std::vector<cv::DMatch> good;
    good.reserve(knn_matches.size());
    for (const auto& m : knn_matches) {
        if (m.size() == 2 && m[0].distance < cfg_.match_ratio_thresh * m[1].distance)
            good.push_back(m[0]);
    }
    return good;
}

std::vector<cv::DMatch> FeatureTracker::filter_by_homography(
    const std::vector<cv::KeyPoint>& kp1,
    const std::vector<cv::KeyPoint>& kp2,
    const std::vector<cv::DMatch>& matches) const
{
    if (matches.size() < 4) return {};

    std::vector<cv::Point2f> p1, p2;
    for (const auto& m : matches) {
        p1.push_back(kp1[m.queryIdx].pt);
        p2.push_back(kp2[m.trainIdx].pt);
    }

    std::vector<uchar> mask;
    cv::findHomography(p1, p2, cv::RANSAC, 3.0, mask);

    std::vector<cv::DMatch> inliers;
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i]) inliers.push_back(matches[i]);

    return inliers;
}

} // namespace vslam