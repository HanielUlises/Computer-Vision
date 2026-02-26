#pragma once

#include "slam_types.hpp"
#include <optional>

namespace vslam {

struct PoseEstimationResult {
    SE3d pose;
    std::vector<int> inlier_indices;
    int num_inliers;
    bool success;
};

class PoseEstimator {
public:
    explicit PoseEstimator(const CameraIntrinsics& K, const SLAMConfig& cfg);

    PoseEstimationResult estimate_essential(
        const Frame::Ptr& ref,
        const Frame::Ptr& cur,
        const std::vector<cv::DMatch>& matches) const;

    PoseEstimationResult estimate_pnp(
        const Frame::Ptr& cur,
        const std::vector<Vec3d>& world_pts,
        const std::vector<cv::Point2f>& image_pts,
        const SE3d& initial_pose = SE3d::Identity()) const;

    std::optional<Vec3d> triangulate(
        const cv::Point2f& pt1, const cv::Point2f& pt2,
        const SE3d& pose1, const SE3d& pose2) const;

    std::vector<Vec3d> triangulate_batch(
        const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2,
        const SE3d& pose1, const SE3d& pose2,
        std::vector<bool>& valid) const;

private:
    CameraIntrinsics K_;
    SLAMConfig       cfg_;

    SE3d recover_pose_from_essential(
        const cv::Mat& E,
        const std::vector<cv::Point2f>& pts1,
        const std::vector<cv::Point2f>& pts2,
        std::vector<uchar>& inlier_mask) const;

    bool is_positive_depth(const Vec3d& p,
                            const SE3d& pose1,
                            const SE3d& pose2) const;
};

} // namespace vslam