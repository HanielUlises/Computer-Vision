#include "feature_detector.h"
#include <opencv2/imgproc.hpp>

FeatureDetector::FeatureDetector(float sigma, int num_octaves, int scales_per_octave)
    : sigma(sigma), num_octaves(num_octaves), scales_per_octave(scales_per_octave) {}

void FeatureDetector::detect(const cv::Mat& image, std::vector<KeyPoint>& keypoints) {
    std::vector<std::vector<cv::Mat>> gaussian_pyramid;
    std::vector<std::vector<cv::Mat>> dog_pyramid;

    build_gaussian_pyramid(image, gaussian_pyramid);
    compute_dog(gaussian_pyramid, dog_pyramid);
    find_extrema(dog_pyramid, keypoints);
}

void FeatureDetector::build_gaussian_pyramid(const cv::Mat& image,
                                            std::vector<std::vector<cv::Mat>>& pyramid) {
    cv::Mat base;
    cv::cvtColor(image, base, cv::COLOR_BGR2GRAY);
    pyramid.resize(num_octaves);

    for (int o = 0; o < num_octaves; ++o) {
        pyramid[o].resize(scales_per_octave + 3);
        float k = std::pow(2.0f, 1.0f / scales_per_octave);
        for (int s = 0; s < scales_per_octave + 3; ++s) {
            float sigma_curr = sigma * std::pow(k, s) * std::pow(2.0f, o);
            cv::Mat blurred;
            int kernel_size = cvRound(sigma_curr * 6) | 1; // Ensure odd
            cv::GaussianBlur(o == 0 && s == 0 ? base : pyramid[o][s - 1], blurred,
                             cv::Size(kernel_size, kernel_size), sigma_curr, sigma_curr);
            pyramid[o][s] = blurred;

            // Downsample for next octave
            if (s == scales_per_octave + 2 && o < num_octaves - 1) {
                cv::Mat downsampled;
                cv::resize(pyramid[o][s], downsampled, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
                pyramid[o + 1][0] = downsampled;
            }
        }
    }
}

void FeatureDetector::compute_dog(const std::vector<std::vector<cv::Mat>>& gaussian_pyramid,
                                  std::vector<std::vector<cv::Mat>>& dog_pyramid) {
    dog_pyramid.resize(num_octaves);
    for (int o = 0; o < num_octaves; ++o) {
        dog_pyramid[o].resize(scales_per_octave + 2);
        for (int s = 0; s < scales_per_octave + 2; ++s) {
            dog_pyramid[o][s] = gaussian_pyramid[o][s + 1] - gaussian_pyramid[o][s];
        }
    }
}

void FeatureDetector::find_extrema(const std::vector<std::vector<cv::Mat>>& dog_pyramid,
                                   std::vector<KeyPoint>& keypoints) {
    for (int o = 0; o < num_octaves; ++o) {
        for (int s = 1; s < scales_per_octave + 1; ++s) {
            const cv::Mat& curr = dog_pyramid[o][s];
            const cv::Mat& prev = dog_pyramid[o][s - 1];
            const cv::Mat& next = dog_pyramid[o][s + 1];

            for (int y = 1; y < curr.rows - 1; ++y) {
                for (int x = 1; x < curr.cols - 1; ++x) {
                    float val = curr.at<float>(y, x);
                    bool is_extrema = true;

                    // Check 3x3x3 neighborhood
                    for (int dy = -1; dy <= 1 && is_extrema; ++dy) {
                        for (int dx = -1; dx <= 1 && is_extrema; ++dx) {
                            if (dy == 0 && dx == 0) continue;
                            if (val <= curr.at<float>(y + dy, x + dx) ||
                                val <= prev.at<float>(y + dy, x + dx) ||
                                val <= next.at<float>(y + dy, x + dx)) {
                                is_extrema = false;
                            }
                        }
                    }

                    if (is_extrema && std::abs(val) > 0.03) {
                        KeyPoint kp;
                        kp.pt = cv::Point2f(x * std::pow(2.0f, o), y * std::pow(2.0f, o));
                        kp.scale = sigma * std::pow(2.0f, (s + o * scales_per_octave) / scales_per_octave);
                        kp.response = std::abs(val);
                        keypoints.push_back(kp);
                    }
                }
            }
        }
    }
}