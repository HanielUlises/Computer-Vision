#pragma once

#include "slam_types.hpp"
#include "map.hpp"
#include <opencv2/viz.hpp>

namespace vslam {

class Visualizer {
public:
    explicit Visualizer(const std::string& window_name = "Visual SLAM");

    void update(const Map::Ptr& map, const Frame::Ptr& current_frame);

    void draw_frame(const Frame::Ptr& frame,
                    const cv::Scalar& color = {0, 255, 0});

    static cv::Mat draw_matches(const Frame::Ptr& ref,
                                 const Frame::Ptr& cur,
                                 const std::vector<cv::DMatch>& matches,
                                 int max_draw = 100);

    static cv::Mat draw_tracked_points(const Frame::Ptr& frame);

    bool is_stopped() const;
    void spin_once(int delay_ms = 1);

private:
    cv::viz::Viz3d window_;
    std::vector<cv::Affine3d> camera_poses_;
    bool stopped_{false};
};

} // namespace vslam