#include <iostream>
#include "dataPath.hpp"
#include <opencv2/opencv.hpp>

cv::Mat load_image(const std::string& path) {
    return cv::imread(path);
}

cv::Mat to_grayscale(const cv::Mat& image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

void display_image(const std::string& window_name, const cv::Mat& image) {
    cv::imshow(window_name, image);
    cv::waitKey(0);
}

std::vector<std::vector<cv::Point>> find_contours(const cv::Mat& image, int mode) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(image, contours, hierarchy, mode, cv::CHAIN_APPROX_SIMPLE);
    std::cout << "Number of contours found = " << contours.size() << std::endl;
    return contours;
}

void draw_all_contours(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours, int thickness) {
    cv::drawContours(image, contours, -1, cv::Scalar(0, 255, 0), thickness);
}

void draw_single_contour(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours, int index, int thickness) {
    cv::drawContours(image, contours, index, cv::Scalar(0, 255, 0), thickness);
}

void mark_centroids(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours) {
    cv::Moments moments;
    for (size_t i = 0; i < contours.size(); i++) {
        moments = cv::moments(contours[i]);
        int x = int(moments.m10 / double(moments.m00));
        int y = int(moments.m01 / double(moments.m00));
        cv::circle(image, cv::Point(x, y), 10, cv::Scalar(255, 0, 0), -1);
    }
}

void mark_contours_with_numbers(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours) {
    cv::Moments moments;
    for (size_t i = 0; i < contours.size(); i++) {
        moments = cv::moments(contours[i]);
        int x = int(moments.m10 / double(moments.m00));
        int y = int(moments.m01 / double(moments.m00));
        cv::circle(image, cv::Point(x, y), 10, cv::Scalar(255, 0, 0), -1);
        cv::putText(image, std::to_string(i + 1), cv::Point(x + 40, y - 10), 
                   cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    }
}

void print_contour_metrics(const std::vector<std::vector<cv::Point>>& contours) {
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        double perimeter = cv::arcLength(contours[i], true);
        std::cout << "Contour #" << i + 1 << " has area = " << area << " and perimeter = " << perimeter << std::endl;
    }
}

void draw_bounding_rects(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours) {
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Rect rect = cv::boundingRect(contours[i]);
        cv::rectangle(image, rect, cv::Scalar(255, 0, 255), 2);
    }
}

void draw_rotated_rects(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours) {
    cv::Mat box_points_2f, box_points_cov;
    for (size_t i = 0; i < contours.size(); i++) {
        cv::RotatedRect rot_rect = cv::minAreaRect(contours[i]);
        cv::boxPoints(rot_rect, box_points_2f);
        box_points_2f.assignTo(box_points_cov, CV_32S);
        cv::polylines(image, box_points_cov, true, cv::Scalar(0, 255, 255), 2);
    }
}

void draw_enclosing_circles(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours) {
    for (size_t i = 0; i < contours.size(); i++) {
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contours[i], center, radius);
        cv::circle(image, center, radius, cv::Scalar(125, 125, 125), 2);
    }
}

void draw_fitted_ellipses(cv::Mat& image, const std::vector<std::vector<cv::Point>>& contours) {
    for (size_t i = 0; i < contours.size(); i++) {
        if (contours[i].size() < 5) continue;
        cv::RotatedRect ellipse_rect = cv::fitEllipse(contours[i]);
        cv::ellipse(image, ellipse_rect, cv::Scalar(255, 0, 125), 2);
    }
}

int main() {
    std::string image_path = DATA_PATH + "images/Contour.png";
    cv::Mat original_image = load_image(image_path);
    cv::Mat image_gray = to_grayscale(original_image);
    
    display_image("image", image_gray);
    
    auto all_contours = find_contours(image_gray, cv::RETR_LIST);
    cv::Mat image = original_image.clone();
    draw_all_contours(image, all_contours, 6);
    display_image("image", image);
    
    auto external_contours = find_contours(image_gray, cv::RETR_EXTERNAL);
    image = original_image.clone();
    draw_all_contours(image, external_contours, 3);
    display_image("image", image);
    
    image = original_image.clone();
    draw_single_contour(image, external_contours, 2, 3);
    display_image("image", image);
    
    image = original_image.clone();
    all_contours = find_contours(image_gray, cv::RETR_LIST);
    draw_all_contours(image, all_contours, 3);
    mark_centroids(image, all_contours);
    display_image("image", image);
    
    mark_contours_with_numbers(image, all_contours);
    cv::Mat image_copy = image.clone();
    display_image("image", image);
    
    print_contour_metrics(all_contours);
    
    image = image_copy.clone();
    draw_bounding_rects(image, all_contours);
    display_image("image", image);
    
    image = image_copy.clone();
    draw_rotated_rects(image, all_contours);
    display_image("image", image);
    
    image = image_copy.clone();
    draw_enclosing_circles(image, all_contours);
    display_image("image", image);
    
    image = image_copy.clone();
    draw_fitted_ellipses(image, all_contours);
    display_image("image", image);
    
    return 0;
}