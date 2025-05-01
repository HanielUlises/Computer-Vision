#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <optional>
#include <string>

namespace geometric_analysis {
struct shape_info {
    std::string type;
    double area;
    double perimeter;
    cv::Point2f centroid;
    std::vector<cv::Point> contour;
};

class geometric_shape_detector {
public:
    explicit geometric_shape_detector(const std::string& image_path);
    std::vector<shape_info> detect_shapes();
    void save_annotated_image(const std::string& output_path);
private:
    cv::Mat preprocess_image();
    std::vector<std::vector<cv::Point>> find_contours(const cv::Mat& edges);
    std::optional<shape_info> analyze_contour(const std::vector<cv::Point>& contour);
    bool is_circle(const std::vector<cv::Point>& contour, double area, double perimeter);
    bool is_triangle(const std::vector<cv::Point>& contour);
    bool is_rectangle(const std::vector<cv::Point>& contour);
    shape_info compute_shape_properties(const std::vector<cv::Point>& contour, const std::string& type);
    cv::Mat input_image;
    cv::Mat processed_image;
};
} // namespace geometric_analysis