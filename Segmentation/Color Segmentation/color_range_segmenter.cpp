#include "color_range_segmenter.hpp"
#include <iostream>

color_range_segmenter::color_range_segmenter(const cv::Scalar& lower_bound,
                                             const cv::Scalar& upper_bound)
    : lower_bound_(lower_bound),
      upper_bound_(upper_bound)
{
}

cv::Mat color_range_segmenter::read_rgb_image(const std::string& image_path,
                                              bool show) const
{
    cv::Mat image = cv::imread(image_path);

    if (image.empty()) {
        std::cerr << "Error loading image: " << image_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    if (show) {
        cv::imshow("rgb_image", image);
    }

    return image;
}

cv::Mat color_range_segmenter::filter_color(const cv::Mat& rgb_image) const
{
    cv::Mat hsv_image;
    cv::cvtColor(rgb_image, hsv_image, cv::COLOR_BGR2HSV);

    cv::Mat mask;
    cv::inRange(hsv_image, lower_bound_, upper_bound_, mask);

    return mask;
}

std::vector<std::vector<cv::Point>>
color_range_segmenter::get_contours(const cv::Mat& binary_image) const
{
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(binary_image.clone(),
                     contours,
                     hierarchy,
                     cv::RETR_EXTERNAL,
                     cv::CHAIN_APPROX_SIMPLE);

    return contours;
}

void color_range_segmenter::draw_contours(
    cv::Mat& rgb_image,
    const std::vector<std::vector<cv::Point>>& contours,
    double min_area) const
{
    cv::Mat debug_image =
        cv::Mat::zeros(rgb_image.size(), CV_8UC3);

    for (const auto& contour : contours)
    {
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);

        if (area < min_area)
            continue;

        cv::drawContours(rgb_image,
                         std::vector<std::vector<cv::Point>>{contour},
                         -1,
                         cv::Scalar(150, 250, 150),
                         1);

        cv::drawContours(debug_image,
                         std::vector<std::vector<cv::Point>>{contour},
                         -1,
                         cv::Scalar(150, 250, 150),
                         1);

        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contour, center, radius);

        cv::Point contour_center =
            get_contour_center(contour);

        cv::circle(rgb_image,
                   contour_center,
                   static_cast<int>(radius),
                   cv::Scalar(0, 0, 255),
                   1);

        cv::circle(debug_image,
                   contour_center,
                   static_cast<int>(radius),
                   cv::Scalar(0, 0, 255),
                   1);

        cv::circle(debug_image,
                   contour_center,
                   5,
                   cv::Scalar(150, 150, 255),
                   -1);

        std::cout << "area: " << area
                  << ", perimeter: " << perimeter
                  << std::endl;
    }

    std::cout << "number_of_contours: "
              << contours.size()
              << std::endl;

    cv::imshow("rgb_image_contours", rgb_image);
    cv::imshow("debug_contours", debug_image);
}

cv::Point color_range_segmenter::get_contour_center(
    const std::vector<cv::Point>& contour) const
{
    cv::Moments m = cv::moments(contour);

    if (m.m00 != 0.0) {
        int cx = static_cast<int>(m.m10 / m.m00);
        int cy = static_cast<int>(m.m01 / m.m00);
        return cv::Point(cx, cy);
    }

    return cv::Point(-1, -1);
}
