#include "frontend.hpp"
#include <iostream>

namespace vslam {

Frontend::Frontend(const CameraIntrinsics& K,
                   const SLAMConfig& cfg,
                   Map::Ptr map)
    : K_(K), cfg_(cfg), map_(std::move(map)),
      state_(TrackingState::NOT_INITIALIZED)
{
    tracker_   = std::make_unique<FeatureTracker>(cfg_);
    estimator_ = std::make_unique<PoseEstimator>(K_, cfg_);
}

bool Frontend::process_frame(Frame::Ptr& frame) {
    frame->id = next_frame_id_++;
    tracker_->detect_and_compute(frame);

    bool ok = false;
    switch (state_) {
        case TrackingState::NOT_INITIALIZED:
            ok = initialize(frame);
            break;
        case TrackingState::INITIALIZING:
        case TrackingState::TRACKING:
            ok = track(frame);
            break;
        case TrackingState::LOST:
            ok = relocalize(frame);
            break;
    }

    last_frame_ = frame;
    return ok;
}

bool Frontend::initialize(Frame::Ptr& frame) {
    if (!last_frame_) {
        state_ = TrackingState::INITIALIZING;
        last_frame_ = frame;
        return true;
    }

    auto matches = tracker_->match(last_frame_, frame);
    if (static_cast<int>(matches.size()) < cfg_.min_inliers) {
        last_frame_ = frame;
        return false;
    }

    double parallax = compute_parallax(last_frame_, frame, matches);
    if (parallax < cfg_.keyframe_parallax) return true;

    auto result = estimator_->estimate_essential(last_frame_, frame, matches);
    if (!result.success) return false;

    frame->pose = result.pose;
    last_frame_->pose = SE3d::Identity();

    std::vector<cv::Point2f> pts1, pts2;
    for (int idx : result.inlier_indices) {
        pts1.push_back(last_frame_->keypoints[matches[idx].queryIdx].pt);
        pts2.push_back(frame->keypoints[matches[idx].trainIdx].pt);
    }

    std::vector<bool> valid;
    auto pts3d = estimator_->triangulate_batch(pts1, pts2,
                                                last_frame_->pose,
                                                frame->pose, valid);

    std::vector<cv::DMatch> inlier_matches;
    for (int idx : result.inlier_indices)
        inlier_matches.push_back(matches[idx]);

    update_landmark_associations(frame, inlier_matches, valid, pts3d);

    last_frame_->is_keyframe = true;
    frame->is_keyframe = true;
    map_->insert_keyframe(last_frame_);
    map_->insert_keyframe(frame);
    last_keyframe_ = frame;

    state_ = TrackingState::TRACKING;
    std::cout << "[SLAM] Initialized with " << result.num_inliers
              << " inliers, " << map_->num_landmarks() << " landmarks.\n";
    return true;
}

bool Frontend::track(Frame::Ptr& frame) {
    frame->pose = last_frame_->pose;

    std::vector<cv::DMatch> matches = tracker_->match(last_frame_, frame);
    if (static_cast<int>(matches.size()) < cfg_.min_inliers) {
        state_ = TrackingState::LOST;
        return false;
    }

    std::vector<Vec3d>       world_pts;
    std::vector<cv::Point2f> image_pts;
    for (const auto& m : matches) {
        auto it = last_frame_->landmark_associations.find(m.queryIdx);
        if (it == last_frame_->landmark_associations.end()) continue;
        auto lm = map_->get_landmark(it->second);
        if (!lm || lm->is_outlier) continue;
        world_pts.push_back(lm->world_position);
        image_pts.push_back(frame->keypoints[m.trainIdx].pt);
    }

    if (static_cast<int>(world_pts.size()) >= 6) {
        auto pnp = estimator_->estimate_pnp(frame, world_pts, image_pts, frame->pose);
        if (pnp.success) {
            frame->pose = pnp.pose;
        } else {
            state_ = TrackingState::LOST;
            return false;
        }
    }

    ++frames_since_keyframe_;
    if (is_keyframe(frame))
        create_keyframe(frame);

    return true;
}

bool Frontend::relocalize(Frame::Ptr& frame) {
    if (!last_keyframe_) return false;

    auto matches = tracker_->match(last_keyframe_, frame);
    if (static_cast<int>(matches.size()) < cfg_.min_inliers) return false;

    auto result = estimator_->estimate_essential(last_keyframe_, frame, matches);
    if (!result.success) return false;

    frame->pose = last_keyframe_->pose * result.pose;
    state_ = TrackingState::TRACKING;
    frames_since_keyframe_ = 0;
    return true;
}

bool Frontend::is_keyframe(const Frame::Ptr& frame) const {
    if (frames_since_keyframe_ < cfg_.keyframe_min_interval) return false;

    if (!last_keyframe_) return true;

    auto matches = tracker_->match(last_keyframe_, frame);
    double parallax = compute_parallax(last_keyframe_, frame, matches);
    return parallax > cfg_.keyframe_parallax;
}

void Frontend::create_keyframe(Frame::Ptr& frame) {
    frame->is_keyframe = true;
    map_->insert_keyframe(frame);
    last_keyframe_ = frame;
    frames_since_keyframe_ = 0;

    auto matches = tracker_->match(last_frame_, frame);
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(last_frame_->keypoints[m.queryIdx].pt);
        pts2.push_back(frame->keypoints[m.trainIdx].pt);
    }

    std::vector<bool> valid;
    auto pts3d = estimator_->triangulate_batch(pts1, pts2,
                                                last_frame_->pose,
                                                frame->pose, valid);
    update_landmark_associations(frame, matches, valid, pts3d);

    map_->clean_map();
    std::cout << "[SLAM] Keyframe #" << frame->id
              << " | landmarks: " << map_->num_landmarks()
              << " | pose: [" << frame->pose.translation().transpose() << "]\n";
}

int Frontend::match_local_map(Frame::Ptr& frame) {
    int count = 0;
    auto active_lms = map_->get_active_landmarks();
    for (const auto& lm : active_lms) {
        Vec3d p_cam = frame->pose * lm->world_position;
        if (p_cam.z() <= 0) continue;
        cv::Point2f proj = K_.project(p_cam);
        if (proj.x < 0 || proj.y < 0) continue;
        ++count;
    }
    return count;
}

void Frontend::update_landmark_associations(
    Frame::Ptr& frame,
    const std::vector<cv::DMatch>& matches,
    const std::vector<bool>& valid_tri,
    const std::vector<Vec3d>& tri_pts)
{
    for (size_t i = 0; i < matches.size(); ++i) {
        if (i >= valid_tri.size() || !valid_tri[i]) continue;
        auto lm = std::make_shared<Landmark>(next_landmark_id_++);
        lm->world_position = tri_pts[i];
        lm->observation_count = 2;

        int kp_ref = matches[i].queryIdx;
        int kp_cur = matches[i].trainIdx;
        lm->observations.emplace_back(last_frame_ ? last_frame_->id : 0, kp_ref);
        lm->observations.emplace_back(frame->id, kp_cur);

        frame->landmark_associations[kp_cur] = lm->id;
        if (last_frame_)
            last_frame_->landmark_associations[kp_ref] = lm->id;

        map_->insert_landmark(lm);
    }
}

double Frontend::compute_parallax(const Frame::Ptr& ref,
                                   const Frame::Ptr& cur,
                                   const std::vector<cv::DMatch>& matches) const
{
    if (matches.empty()) return 0.0;
    double total = 0.0;
    for (const auto& m : matches) {
        cv::Point2f d = ref->keypoints[m.queryIdx].pt
                      - cur->keypoints[m.trainIdx].pt;
        total += std::sqrt(d.x * d.x + d.y * d.y);
    }
    return total / static_cast<double>(matches.size());
}

} // namespace vslam