#include "slam_system.hpp"
#include <fstream>
#include <iomanip>

namespace vslam {

SLAMSystem::SLAMSystem(const CameraIntrinsics& K, const SLAMConfig& cfg)
    : K_(K), cfg_(cfg)
{
    map_      = std::make_shared<Map>();
    frontend_ = std::make_shared<Frontend>(K_, cfg_, map_);
}

bool SLAMSystem::process_image(const cv::Mat& img, double timestamp) {
    auto frame = std::make_shared<Frame>(-1, timestamp);
    frame->image = img;

    bool ok = frontend_->process_frame(frame);
    if (ok) trajectory_.emplace_back(timestamp, frame->pose);
    return ok;
}

SE3d SLAMSystem::current_pose() const {
    return trajectory_.empty() ? SE3d::Identity() : trajectory_.back().second;
}

size_t SLAMSystem::num_landmarks() const { return map_->num_landmarks(); }
size_t SLAMSystem::num_keyframes() const { return map_->num_keyframes(); }

void SLAMSystem::enable_visualization(bool enable) {
    visualize_ = enable;
    if (enable && !viz_)
        viz_ = std::make_unique<Visualizer>("Visual SLAM");
}

void SLAMSystem::save_trajectory(const std::string& path) const {
    std::ofstream f(path);
    f << std::fixed << std::setprecision(6);
    for (const auto& [ts, pose] : trajectory_) {
        const auto& t = pose.translation();
        Eigen::Quaterniond q(pose.rotation());
        f << ts << " "
          << t.x() << " " << t.y() << " " << t.z() << " "
          << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << "\n";
    }
}

} // namespace vslam