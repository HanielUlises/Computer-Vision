#include "line_detector.h"
#include <algorithm>

LineDetector::LineDetector(float distance_threshold, int min_points)
    : distance_threshold(distance_threshold), min_points(min_points) {}

void LineDetector::detect(const std::vector<KeyPoint>& keypoints, std::vector<Line>& lines) {
    std::vector<std::vector<KeyPoint>> clusters;

    // Cluster keypoints by proximity
    for (const auto& kp : keypoints) {
        bool added = false;
        for (auto& cluster : clusters) {
            if (!cluster.empty() &&
                cv::norm(kp.pt - cluster.back().pt) < distance_threshold) {
                cluster.push_back(kp);
                added = true;
                break;
            }
        }
        if (!added) {
            clusters.push_back({kp});
        }
    }

    // Fit lines to clusters
    for (const auto& cluster : clusters) {
        if (cluster.size() < min_points) continue;

        // Find extremities
        auto minmax_x = std::minmax_element(
            cluster.begin(), cluster.end(),
            [](const KeyPoint& a, const KeyPoint& b) { return a.pt.x < b.pt.x; });

        Line line;
        line.start = minmax_x.first->pt;
        line.end = minmax_x.second->pt;
        lines.push_back(line);
    }
}

float LineDetector::distance_to_line(const cv::Point2f& p, const cv::Point2f& a,
                                    const cv::Point2f& b) {
    float length_sq = cv::norm(b - a);
    if (length_sq == 0) return cv::norm(p - a);

    float t = std::max(0.0f, std::min(1.0f, ((p - a).dot(b - a)) / length_sq));
    cv::Point2f projection = a + t * (b - a);
    return cv::norm(p - projection);
}