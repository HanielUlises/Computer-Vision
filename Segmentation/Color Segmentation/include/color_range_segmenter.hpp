#ifndef COLOR_RANGE_SEGMENTER_HPP
#define COLOR_RANGE_SEGMENTER_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class color_range_segmenter
{
public:
    color_range_segmenter(const cv::Scalar& lower_bound,
                          const cv::Scalar& upper_bound);

    cv::Mat read_rgb_image(const std::string& image_path, bool show = false) const;

    cv::Mat filter_color(const cv::Mat& rgb_image) const;

    std::vector<std::vector<cv::Point>>
    get_contours(const cv::Mat& binary_image) const;

    void draw_contours(cv::Mat& rgb_image,
                       const std::vector<std::vector<cv::Point>>& contours,
                       double min_area = 100.0) const;

private:
    cv::Point get_contour_center(const std::vector<cv::Point>& contour) const;

    cv::Scalar lower_bound_;
    cv::Scalar upper_bound_;
};

#endif
