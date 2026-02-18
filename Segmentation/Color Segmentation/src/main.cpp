#include "color_range_segmenter.hpp"

int main()
{
    cv::Scalar yellow_lower(30, 150, 100);
    cv::Scalar yellow_upper(50, 255, 255);

    color_range_segmenter segmenter(yellow_lower, yellow_upper);

    cv::Mat image =
        segmenter.read_rgb_image("Images/data/tennisball05.jpg", true);

    cv::Mat mask = segmenter.filter_color(image);

    std::vector<std::vector<cv::Point>> contours =
        segmenter.get_contours(mask);

    segmenter.draw_contours(image, contours);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
