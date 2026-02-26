#pragma once

#include "slam_types.hpp"
#include "frontend.hpp"
#include "map.hpp"
#include "visualizer.hpp"

namespace vslam {

class SLAMSystem {
public:
    SLAMSystem(const CameraIntrinsics& K, const SLAMConfig& cfg);

    bool process_image(const cv::Mat& img, double timestamp = 0.0);

    SE3d   current_pose()    const;
    size_t num_landmarks()   const;
    size_t num_keyframes()   const;

    Map::Ptr    map()     const { return map_; }
    TrackingState state() const { return frontend_->state(); }

    void enable_visualization(bool enable);

    void save_trajectory(const std::string& path) const;

private:
    CameraIntrinsics K_;
    SLAMConfig       cfg_;

    Map::Ptr              map_;
    Frontend::Ptr         frontend_;
    std::unique_ptr<Visualizer> viz_;

    std::vector<std::pair<double, SE3d>> trajectory_;
    bool visualize_{false};
};

} // namespace vslam