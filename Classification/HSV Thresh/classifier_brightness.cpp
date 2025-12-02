#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <vector>
#include <string>
#include <execution>

struct classifier_config
{
    float brightness_threshold = 110.0f; 
};



float compute_average_brightness(const cv::Mat& bgr_image)
{
    if (bgr_image.empty())
    {
        throw std::runtime_error("compute_average_brightness: empty input image");
    }

    cv::Mat hsv_image;
    cv::cvtColor(bgr_image, hsv_image, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv_image, hsv_channels);

    const cv::Mat& v_channel = hsv_channels[2];

    std::vector<uint8_t> v_values;
    v_values.assign(v_channel.datastart, v_channel.dataend);

    auto sum_v = std::reduce(v_values.begin(), v_values.end(), 0ull);

    float total_pixels = static_cast<float>(v_values.size());
    float avg_brightness = static_cast<float>(sum_v) / total_pixels;

    return avg_brightness;
}


std::string classify_day_night(float brightness, const classifier_config& cfg)
{
    return (brightness >= cfg.brightness_threshold) ? "day" : "night";
}


void print_usage(const std::string& exe_name)
{
    std::cout << "usage:\n"
              << "  " << exe_name << " <image_path> [brightness_threshold]\n\n"
              << "example:\n"
              << "  " << exe_name << " image.jpg\n"
              << "  " << exe_name << " image.jpg 120\n";
}


int main(int argc, char** argv)
{
    if (argc < 2)
    {
        print_usage(argv[0]);
        return 1;
    }

    std::string image_path = argv[1];

    classifier_config cfg;
    if (argc >= 3)
    {
        try
        {
            cfg.brightness_threshold = std::stof(argv[2]);
        }
        catch (...)
        {
            std::cerr << "error: invalid brightness threshold value\n";
            return 1;
        }
    }

    // Load image
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty())
    {
        std::cerr << "error: could not load image '" << image_path << "'\n";
        return 1;
    }

    try
    {
        float avg_brightness = compute_average_brightness(image);
        std::string label = classify_day_night(avg_brightness, cfg);

        std::cout << "image:                 " << image_path << "\n";
        std::cout << "average brightness:    " << avg_brightness << "\n";
        std::cout << "threshold:             " << cfg.brightness_threshold << "\n";
        std::cout << "classification:        " << label << "\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
