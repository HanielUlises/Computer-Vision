#include "geometric_shape_detector.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <ranges>
#include <cmath>

namespace geometric_analysis {
geometric_shape_detector::geometric_shape_detector(const std::string& image_path) {
    input_image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (input_image.empty()) {
        throw std::runtime_error("failed_to_load_image");
    }
}

cv::Mat geometric_shape_detector::preprocess_image() {
    cv::Mat gray;
    cv::cvtColor(input_image, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);
    cv::Mat edges;
    cv::Canny(gray, edges, 50, 150);
    return edges;
}

std::vector<std::vector<cv::Point>> geometric_shape_detector::find_contours(const cv::Mat& edges) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

bool geometric_shape_detector::is_circle(const std::vector<cv::Point>& contour, double area, double perimeter) {
    double circularity = 4 * CV_PI * area / (perimeter * perimeter);
    return circularity > 0.8;
}

bool geometric_shape_detector::is_triangle(const std::vector<cv::Point>& contour) {
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.04 * cv::arcLength(contour, true), true);
    return approx.size() == 3;
}

bool geometric_shape_detector::is_rectangle(const std::vector<cv::Point>& contour) {
    std::vector<cv::Point> approx;
    cv::approxPolyDP(contour, approx, 0.04 * cv::arcLength(contour, true), true);
    if (approx.size() != 4) return false;
    cv::Rect rect = cv::boundingRect(contour);
    double aspect_ratio = static_cast<double>(rect.width) / rect.height;
    return aspect_ratio > 0.8 && aspect_ratio < 1.2;
}

shape_info geometric_shape_detector::compute_shape_properties(const std::vector<cv::Point>& contour, const std::string& type) {
    shape_info info;
    info.type = type;
    info.area = cv::contourArea(contour);
    info.perimeter = cv::arcLength(contour, true);
    cv::Moments moments = cv::moments(contour);
    info.centroid = cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00);
    info.contour = contour;
    return info;
}

std::optional<shape_info> geometric_shape_detector::analyze_contour(const std::vector<cv::Point>& contour) {
    if (contour.size() < 10 || cv::contourArea(contour) < 100) {
        return std::nullopt;
    }
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    if (is_circle(contour, area, perimeter)) {
        return compute_shape_properties(contour, "circle");
    }
    if (is_triangle(contour)) {
        return compute_shape_properties(contour, "triangle");
    }
    if (is_rectangle(contour)) {
        return compute_shape_properties(contour, "rectangle");
    }
    return std::nullopt;
}

std::vector<shape_info> geometric_shape_detector::detect_shapes() {
    processed_image = preprocess_image();
    auto contours = find_contours(processed_image);
    std::vector<shape_info> shapes;
    for (const auto& contour : contours) {
        if (auto shape = analyze_contour(contour)) {
            shapes.push_back(*shape);
        }
    }
    return shapes;
}

void geometric_shape_detector::save_annotated_image(const std::string& output_path) {
    cv::Mat annotated = input_image.clone();
    auto shapes = detect_shapes();
    for (const auto& shape : shapes) {
        cv::Scalar color = shape.type == "circle" ? cv::Scalar(0, 255, 0) :
                          shape.type == "triangle" ? cv::Scalar(255, 0, 0) :
                          cv::Scalar(0, 0, 255);
        cv::drawContours(annotated, std::vector<std::vector<cv::Point>>{shape.contour}, -1, color, 2);
        cv::circle(annotated, shape.centroid, 5, color, -1);
        cv::putText(annotated, shape.type, shape.centroid + cv::Point2f(10, 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
    cv::imwrite(output_path, annotated);
}
} // namespace geometric_analysis