#include "visualizer.hpp"
#include <opencv2/viz/vizcore.hpp>

namespace vslam {

Visualizer::Visualizer(const std::string& window_name)
    : window_(window_name)
{
    window_.setBackgroundColor(cv::viz::Color::black());
    window_.showWidget("world_axes", cv::viz::WCoordinateSystem(0.3));
}

void Visualizer::update(const Map::Ptr& map, const Frame::Ptr& current_frame) {
    auto lms = map->get_active_landmarks();
    std::vector<cv::Point3f> cloud;
    cloud.reserve(lms.size());
    for (const auto& lm : lms)
        cloud.emplace_back(static_cast<float>(lm->world_position.x()),
                           static_cast<float>(lm->world_position.y()),
                           static_cast<float>(lm->world_position.z()));

    if (!cloud.empty()) {
        cv::viz::WCloud cloud_widget(cloud, cv::viz::Color::green());
        cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 2.0);
        window_.showWidget("point_cloud", cloud_widget);
    }

    draw_frame(current_frame);

    if (!camera_poses_.empty()) {
        cv::viz::WPolyLine traj(camera_poses_, cv::viz::Color::yellow());
        window_.showWidget("trajectory", traj);
    }

    window_.spinOnce(1, true);
}

void Visualizer::draw_frame(const Frame::Ptr& frame, const cv::Scalar& color) {
    Eigen::Matrix4d T = frame->pose.matrix();
    cv::Matx33d R;
    cv::Vec3d   t;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) R(i, j) = T(i, j);
        t[i] = T(i, 3);
    }
    cv::Affine3d affine(R, t);
    camera_poses_.push_back(affine);

    cv::viz::WCameraPosition cam_frustum(cv::Matx33d::eye(), 0.1,
                                          cv::viz::Color(color[0], color[1], color[2]));
    window_.showWidget("cam_" + std::to_string(frame->id), cam_frustum, affine);
}

cv::Mat Visualizer::draw_matches(const Frame::Ptr& ref,
                                  const Frame::Ptr& cur,
                                  const std::vector<cv::DMatch>& matches,
                                  int max_draw)
{
    std::vector<cv::DMatch> subset(matches.begin(),
                                   matches.begin() + std::min<int>(max_draw, matches.size()));
    cv::Mat out;
    cv::drawMatches(ref->image, ref->keypoints,
                    cur->image, cur->keypoints,
                    subset, out,
                    cv::Scalar::all(-1), cv::Scalar::all(-1),
                    {}, cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    return out;
}

cv::Mat Visualizer::draw_tracked_points(const Frame::Ptr& frame) {
    cv::Mat out = frame->image.clone();
    for (const auto& [kp_idx, lm_id] : frame->landmark_associations) {
        if (kp_idx >= static_cast<int>(frame->keypoints.size())) continue;
        cv::circle(out, frame->keypoints[kp_idx].pt, 4,
                   cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
    }
    for (const auto& kp : frame->keypoints)
        cv::circle(out, kp.pt, 2, cv::Scalar(128, 128, 128), -1, cv::LINE_AA);

    cv::putText(out, "Tracked: " + std::to_string(frame->landmark_associations.size()),
                {10, 25}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
    return out;
}

bool Visualizer::is_stopped() const { return window_.wasStopped(); }
void Visualizer::spin_once(int delay_ms) { window_.spinOnce(delay_ms, true); }

} // namespace vslam