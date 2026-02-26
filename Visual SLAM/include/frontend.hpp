#pragma once

#include "slam_types.hpp"
#include "feature_tracker.hpp"
#include "pose_estimator.hpp"
#include "map.hpp"
#include <atomic>

namespace vslam {

enum class TrackingState {
    NOT_INITIALIZED,
    INITIALIZING,
    TRACKING,
    LOST
};

class Frontend {
public:
    using Ptr = std::shared_ptr<Frontend>;

    Frontend(const CameraIntrinsics& K,
             const SLAMConfig& cfg,
             Map::Ptr map);

    bool process_frame(Frame::Ptr& frame);

    TrackingState state() const { return state_; }
    Frame::Ptr    last_keyframe() const { return last_keyframe_; }

private:
    bool initialize(Frame::Ptr& frame);
    bool track(Frame::Ptr& frame);
    bool relocalize(Frame::Ptr& frame);

    bool is_keyframe(const Frame::Ptr& frame) const;
    void create_keyframe(Frame::Ptr& frame);

    int  match_local_map(Frame::Ptr& frame);
    void update_landmark_associations(Frame::Ptr& frame,
                                       const std::vector<cv::DMatch>& matches,
                                       const std::vector<bool>& valid_tri,
                                       const std::vector<Vec3d>& tri_pts);

    double compute_parallax(const Frame::Ptr& ref,
                             const Frame::Ptr& cur,
                             const std::vector<cv::DMatch>& matches) const;

    CameraIntrinsics K_;
    SLAMConfig       cfg_;
    Map::Ptr         map_;

    std::unique_ptr<FeatureTracker> tracker_;
    std::unique_ptr<PoseEstimator>  estimator_;

    Frame::Ptr       last_frame_;
    Frame::Ptr       last_keyframe_;
    TrackingState    state_;

    std::atomic<int> next_frame_id_{0};
    std::atomic<int> next_landmark_id_{0};
    int              frames_since_keyframe_{0};
};

} // namespace vslam