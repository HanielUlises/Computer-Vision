#ifndef LINE_DETECTOR_H
#define LINE_DETECTOR_H

#include "feature_detector.h"
#include <vector>

struct Line {
    cv::Point2f start;
    cv::Point2f end;
};

class LineDetector {
public:
    LineDetector(float distance_threshold = 10.0f, int min_points = 5);
    void detect(const std::vector<KeyPoint>& keypoints, std::vector<Line>& lines);

private:
    float distance_threshold;
    int min_points;

    float distance_to_line(const cv::Point2f& p, const cv::Point2f& a, const cv::Point2f& b);
};

#endif // LINE_DETECTOR_H