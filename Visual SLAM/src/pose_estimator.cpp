#include "pose_estimator.hpp"
#include <opencv2/calib3d.hpp>

namespace vslam {

PoseEstimator::PoseEstimator(const CameraIntrinsics& K, const SLAMConfig& cfg)
    : K_(K), cfg_(cfg) {}

PoseEstimationResult PoseEstimator::estimate_essential(
    const Frame::Ptr& ref,
    const Frame::Ptr& cur,
    const std::vector<cv::DMatch>& matches) const
{
    if (static_cast<int>(matches.size()) < cfg_.min_inliers)
        return {{}, {}, 0, false};

    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (const auto& m : matches) {
        pts1.push_back(ref->keypoints[m.queryIdx].pt);
        pts2.push_back(cur->keypoints[m.trainIdx].pt);
    }

    std::vector<uchar> mask;
    cv::Mat E = cv::findEssentialMat(pts1, pts2, K_.K,
                                      cv::RANSAC, 0.999,
                                      cfg_.reproj_error_thresh, mask);
    if (E.empty()) return {{}, {}, 0, false};

    SE3d pose = recover_pose_from_essential(E, pts1, pts2, mask);

    std::vector<int> inliers;
    for (size_t i = 0; i < mask.size(); ++i)
        if (mask[i]) inliers.push_back(static_cast<int>(i));

    if (static_cast<int>(inliers.size()) < cfg_.min_inliers)
        return {{}, {}, 0, false};

    return {pose, inliers, static_cast<int>(inliers.size()), true};
}

PoseEstimationResult PoseEstimator::estimate_pnp(
    const Frame::Ptr& cur,
    const std::vector<Vec3d>& world_pts,
    const std::vector<cv::Point2f>& image_pts,
    const SE3d& initial_pose) const
{
    if (static_cast<int>(world_pts.size()) < 4)
        return {{}, {}, 0, false};

    std::vector<cv::Point3f> obj_pts;
    obj_pts.reserve(world_pts.size());
    for (const auto& p : world_pts)
        obj_pts.emplace_back(static_cast<float>(p.x()),
                              static_cast<float>(p.y()),
                              static_cast<float>(p.z()));

    Eigen::AngleAxisd aa(initial_pose.rotation());
    cv::Mat rvec(3, 1, CV_64F), tvec(3, 1, CV_64F);
    auto axis = aa.axis() * aa.angle();
    rvec.at<double>(0) = axis.x();
    rvec.at<double>(1) = axis.y();
    rvec.at<double>(2) = axis.z();
    tvec.at<double>(0) = initial_pose.translation().x();
    tvec.at<double>(1) = initial_pose.translation().y();
    tvec.at<double>(2) = initial_pose.translation().z();

    std::vector<int> inlier_idx;
    bool ok = cv::solvePnPRansac(
        obj_pts, image_pts, K_.K, K_.dist_coeffs,
        rvec, tvec, true, 100,
        static_cast<float>(cfg_.reproj_error_thresh),
        0.999, inlier_idx, cv::SOLVEPNP_ITERATIVE
    );

    if (!ok || static_cast<int>(inlier_idx.size()) < cfg_.min_inliers)
        return {{}, {}, 0, false};

    cv::Mat R;
    cv::Rodrigues(rvec, R);
    SE3d pose = SE3d::Identity();
    Eigen::Matrix3d Reig;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Reig(i, j) = R.at<double>(i, j);
    pose.linear() = Reig;
    pose.translation() = {tvec.at<double>(0),
                           tvec.at<double>(1),
                           tvec.at<double>(2)};

    return {pose, inlier_idx, static_cast<int>(inlier_idx.size()), true};
}

std::optional<Vec3d> PoseEstimator::triangulate(
    const cv::Point2f& pt1, const cv::Point2f& pt2,
    const SE3d& pose1, const SE3d& pose2) const
{
    auto build_proj = [&](const SE3d& pose) -> cv::Mat {
        cv::Mat P(3, 4, CV_64F);
        Eigen::Matrix<double, 3, 4> Rt;
        Rt.leftCols(3) = pose.rotation();
        Rt.rightCols(1) = pose.translation();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 4; ++c)
                P.at<double>(r, c) = Rt(r, c);
        return cv::Mat(K_.K) * P;
    };

    cv::Mat P1 = build_proj(pose1);
    cv::Mat P2 = build_proj(pose2);

    std::vector<cv::Point2f> p1v = {pt1}, p2v = {pt2};
    cv::Mat pts4D;
    cv::triangulatePoints(P1, P2, p1v, p2v, pts4D);

    double w = pts4D.at<double>(3, 0);
    if (std::abs(w) < 1e-8) return std::nullopt;

    Vec3d p(pts4D.at<double>(0, 0) / w,
            pts4D.at<double>(1, 0) / w,
            pts4D.at<double>(2, 0) / w);

    if (!is_positive_depth(p, pose1, pose2)) return std::nullopt;

    return p;
}

std::vector<Vec3d> PoseEstimator::triangulate_batch(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    const SE3d& pose1, const SE3d& pose2,
    std::vector<bool>& valid) const
{
    valid.assign(pts1.size(), false);
    std::vector<Vec3d> result(pts1.size());
    for (size_t i = 0; i < pts1.size(); ++i) {
        auto p = triangulate(pts1[i], pts2[i], pose1, pose2);
        if (p) {
            result[i] = *p;
            valid[i] = true;
        }
    }
    return result;
}

SE3d PoseEstimator::recover_pose_from_essential(
    const cv::Mat& E,
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2,
    std::vector<uchar>& mask) const
{
    cv::Mat R, t;
    cv::recoverPose(E, pts1, pts2, K_.K, R, t, mask);

    SE3d pose = SE3d::Identity();
    Eigen::Matrix3d Reig;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Reig(i, j) = R.at<double>(i, j);
    pose.linear() = Reig;
    pose.translation() = {t.at<double>(0), t.at<double>(1), t.at<double>(2)};
    return pose;
}

bool PoseEstimator::is_positive_depth(const Vec3d& p,
                                       const SE3d& pose1,
                                       const SE3d& pose2) const
{
    Vec3d p_cam1 = pose1 * p;
    Vec3d p_cam2 = pose2 * p;
    return p_cam1.z() > 0 && p_cam2.z() > 0;
}

} // namespace vslam