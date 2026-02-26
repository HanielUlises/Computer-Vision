#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <memory>
#include <unordered_map>
#include <atomic>

namespace vslam {

using SE3d = Eigen::Isometry3d;
using Vec3d = Eigen::Vector3d;
using Mat3d = Eigen::Matrix3d;

struct KeyPoint3D {
    Vec3d position;
    cv::Mat descriptor;
    int id;
    int observation_count;
    bool is_outlier;

    KeyPoint3D() : id(-1), observation_count(0), is_outlier(false) {}
};

struct Frame {
    using Ptr = std::shared_ptr<Frame>;

    int id;
    double timestamp;
    cv::Mat image;
    cv::Mat depth;

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    SE3d pose;
    bool is_keyframe;

    std::unordered_map<int, int> landmark_associations; // keypoint_idx -> landmark_id

    explicit Frame(int id, double ts = 0.0)
        : id(id), timestamp(ts), is_keyframe(false) {}
};

struct Landmark {
    using Ptr = std::shared_ptr<Landmark>;

    int id;
    Vec3d world_position;
    cv::Mat descriptor;
    bool is_outlier;
    int observation_count;

    std::vector<std::pair<int, int>> observations; // {frame_id, keypoint_idx}

    explicit Landmark(int id) : id(id), is_outlier(false), observation_count(0) {}
};

struct CameraIntrinsics {
    double fx, fy, cx, cy;
    double baseline; // for stereo; 0 for mono
    cv::Mat dist_coeffs;
    cv::Mat K;

    CameraIntrinsics() : fx(0), fy(0), cx(0), cy(0), baseline(0) {}

    CameraIntrinsics(double fx, double fy, double cx, double cy,
                     const cv::Mat& dist = cv::Mat())
        : fx(fx), fy(fy), cx(cx), cy(cy), baseline(0), dist_coeffs(dist) {
        K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
    }

    cv::Point2f project(const Vec3d& p) const {
        return {static_cast<float>(fx * p.x() / p.z() + cx),
                static_cast<float>(fy * p.y() / p.z() + cy)};
    }

    Vec3d unproject(const cv::Point2f& pt, double depth = 1.0) const {
        return {(pt.x - cx) / fx * depth,
                (pt.y - cy) / fy * depth,
                depth};
    }
};

struct SLAMConfig {
    int num_features           = 500;
    float scale_factor         = 1.2f;
    int num_pyramid_levels     = 8;
    int fast_threshold         = 20;
    float match_ratio_thresh   = 0.75f;
    int min_inliers            = 20;
    int keyframe_min_interval  = 5;    // frames
    float keyframe_parallax    = 30.0f; // pixels
    int local_map_window       = 10;   // keyframes
    bool use_loop_closure      = false;
    double reproj_error_thresh = 4.0;
};

} // namespace vslam