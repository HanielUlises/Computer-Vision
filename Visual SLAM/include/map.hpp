#pragma once

#include "slam_types.hpp"
#include <mutex>
#include <unordered_map>

namespace vslam {

class Map {
public:
    using Ptr = std::shared_ptr<Map>;

    void insert_keyframe(const Frame::Ptr& kf);
    void insert_landmark(const Landmark::Ptr& lm);

    void remove_outlier_landmarks();

    Frame::Ptr get_keyframe(int id) const;
    Landmark::Ptr get_landmark(int id) const;

    std::vector<Frame::Ptr> get_active_keyframes(int window = 10) const;
    std::vector<Landmark::Ptr> get_active_landmarks() const;

    size_t num_keyframes() const;
    size_t num_landmarks() const;

    void clean_map();

private:
    mutable std::mutex map_mutex_;
    std::unordered_map<int, Frame::Ptr>    keyframes_;
    std::unordered_map<int, Landmark::Ptr> landmarks_;
};

} // namespace vslam