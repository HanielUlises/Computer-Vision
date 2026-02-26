#include "map.hpp"
#include <algorithm>

namespace vslam {

void Map::insert_keyframe(const Frame::Ptr& kf) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    keyframes_[kf->id] = kf;
}

void Map::insert_landmark(const Landmark::Ptr& lm) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    landmarks_[lm->id] = lm;
}

void Map::remove_outlier_landmarks() {
    std::lock_guard<std::mutex> lock(map_mutex_);
    for (auto it = landmarks_.begin(); it != landmarks_.end(); ) {
        if (it->second->is_outlier)
            it = landmarks_.erase(it);
        else
            ++it;
    }
}

Frame::Ptr Map::get_keyframe(int id) const {
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = keyframes_.find(id);
    return it != keyframes_.end() ? it->second : nullptr;
}

Landmark::Ptr Map::get_landmark(int id) const {
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = landmarks_.find(id);
    return it != landmarks_.end() ? it->second : nullptr;
}

std::vector<Frame::Ptr> Map::get_active_keyframes(int window) const {
    std::lock_guard<std::mutex> lock(map_mutex_);
    std::vector<Frame::Ptr> kfs;
    kfs.reserve(keyframes_.size());
    for (const auto& [id, kf] : keyframes_)
        kfs.push_back(kf);

    std::sort(kfs.begin(), kfs.end(),
              [](const Frame::Ptr& a, const Frame::Ptr& b) {
                  return a->id > b->id;
              });

    if (static_cast<int>(kfs.size()) > window)
        kfs.resize(window);
    return kfs;
}

std::vector<Landmark::Ptr> Map::get_active_landmarks() const {
    std::lock_guard<std::mutex> lock(map_mutex_);
    std::vector<Landmark::Ptr> lms;
    lms.reserve(landmarks_.size());
    for (const auto& [id, lm] : landmarks_)
        if (!lm->is_outlier) lms.push_back(lm);
    return lms;
}

size_t Map::num_keyframes() const {
    std::lock_guard<std::mutex> lock(map_mutex_);
    return keyframes_.size();
}

size_t Map::num_landmarks() const {
    std::lock_guard<std::mutex> lock(map_mutex_);
    return landmarks_.size();
}

void Map::clean_map() {
    std::lock_guard<std::mutex> lock(map_mutex_);
    for (auto it = landmarks_.begin(); it != landmarks_.end(); ) {
        if (it->second->is_outlier || it->second->observation_count < 2)
            it = landmarks_.erase(it);
        else
            ++it;
    }
}

} // namespace vslam