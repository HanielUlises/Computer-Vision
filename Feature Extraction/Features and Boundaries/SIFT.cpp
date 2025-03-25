#include "sift.h"
#include <fstream>
#include <iostream>

#include <opencv2/imgproc.hpp> 
#include <opencv2/ml.hpp>  

/* sift::sift() {
    sift_detector = cv::xfeatures2d::SIFT::create();
} */

void sift::extract_to_csv(const std::vector<cv::Mat>& images, const std::string& output_csv) {
    std::vector<std::vector<cv::KeyPoint>> all_keypoints;
    std::vector<cv::Mat> all_descriptors;

    for (const auto& image : images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;

        detect_and_compute(image, keypoints, descriptors);

        all_keypoints.push_back(keypoints);
        all_descriptors.push_back(descriptors);

        std::cout << "extracted " << keypoints.size() << " keypoints from an image" << std::endl;
    }

    log_features(all_keypoints);
    write_csv(all_keypoints, all_descriptors, output_csv);
}

void sift::detect_and_compute(const cv::Mat& image, 
                              std::vector<cv::KeyPoint>& keypoints, 
                              cv::Mat& descriptors) {
    if (image.empty() || image.channels() != 1) {
        std::cerr << "invalid input image for detect_and_compute" << std::endl;
        return;
    }

    // Preprocess: Blur to reduce noise
    cv::Mat blurred;
    cv::GaussianBlur(image, blurred, cv::Size(5, 5), 1.0);

    // Detect keypoints: Simple grid-based intensity thresholding
    keypoints.clear();
    for (int y = 10; y < blurred.rows - 10; y += 10) {
        for (int x = 10; x < blurred.cols - 10; x += 10) {
            float response = blurred.at<uchar>(y, x);
            if (response > 100) { // Arbitrary threshold
                keypoints.emplace_back(cv::KeyPoint(x, y, 5.0, -1, response));
            }
        }
    }

    // Compute descriptors: 16D patch-based
    descriptors = cv::Mat(keypoints.size(), 16, CV_32F);
    for (size_t i = 0; i < keypoints.size(); ++i) {
        float x = keypoints[i].pt.x;
        float y = keypoints[i].pt.y;
        int patch_size = 4;
        int desc_idx = 0;

        for (int dy = -patch_size; dy <= patch_size; dy += 2) {
            for (int dx = -patch_size; dx <= patch_size; dx += 2) {
                int px = static_cast<int>(x + dx);
                int py = static_cast<int>(y + dy);
                if (px >= 0 && px < image.cols && py >= 0 && py < image.rows) {
                    descriptors.at<float>(i, desc_idx) = static_cast<float>(image.at<uchar>(py, px));
                } else {
                    descriptors.at<float>(i, desc_idx) = 0.0f;
                }
                desc_idx++;
            }
        }
    }
}

void sift::build_vocabulary(const std::vector<cv::Mat>& images, int cluster_count) {
    // Collecting all descriptors
    cv::Mat all_descriptors;
    for (const auto& image : images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detect_and_compute(image, keypoints, descriptors);
        if (!descriptors.empty()) {
            all_descriptors.push_back(descriptors);
        }
    }

    if (all_descriptors.empty()) {
        std::cerr << "no descriptors found for vocabulary building" << std::endl;
        return;
    }

    // Run KMeans
    cv::Mat labels;
    vocabulary = cv::Mat();
    cv::kmeans(all_descriptors, cluster_count, labels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               3, cv::KMEANS_PP_CENTERS, vocabulary);

    std::cout << "vocabulary built with " << cluster_count << " visual words" << std::endl;
}

std::vector<cv::Mat> sift::compute_bow_histograms(const std::vector<cv::Mat>& images) {
    if (vocabulary.empty()) {
        std::cerr << "vocabulary not initialized, call build_vocabulary first" << std::endl;
        return {};
    }

    std::vector<cv::Mat> histograms;
    cv::Ptr<cv::ml::KNearest> knn = cv::ml::KNearest::create();
    knn->train(vocabulary, cv::ml::ROW_SAMPLE);

    for (const auto& image : images) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        detect_and_compute(image, keypoints, descriptors);

        if (descriptors.empty()) {
            histograms.push_back(cv::Mat::zeros(1, vocabulary.rows, CV_32F));
            continue;
        }

        // We assign descriptors to nearest visual word
        cv::Mat histogram = cv::Mat::zeros(1, vocabulary.rows, CV_32F);
        cv::Mat labels;
        knn->findNearest(descriptors, 1, labels);

        for (int i = 0; i < labels.rows; ++i) {
            int cluster_idx = labels.at<int>(i);
            histogram.at<float>(0, cluster_idx) += 1.0f;
        }

        // (L1 norm)
        cv::normalize(histogram, histogram, 1.0, 0.0, cv::NORM_L1);
        histograms.push_back(histogram.clone());
    }

    return histograms;
}

void sift::write_csv(const std::vector<std::vector<cv::KeyPoint>>& keypoints, 
                     const std::vector<cv::Mat>& descriptors, 
                     const std::string& output_csv) {
    std::ofstream csv_file(output_csv);

    if (!csv_file.is_open()) {
        std::cerr << "could not open the output csv file: " << output_csv << std::endl;
        return;
    }

    for (size_t i = 0; i < keypoints.size(); ++i) {
        for (size_t j = 0; j < keypoints[i].size(); ++j) {
            const cv::KeyPoint& kp = keypoints[i][j];
            const cv::Mat& desc = descriptors[i].row(j);

            csv_file << kp.pt.x << "," << kp.pt.y << "," << kp.size << "," << kp.angle << ","
                     << kp.response << "," << kp.octave << "," << kp.class_id;

            for (int k = 0; k < desc.cols; ++k) {
                csv_file << "," << desc.at<float>(k);
            }
            csv_file << "\n";
        }
    }

    csv_file.close();
    std::cout << "features written to " << output_csv << std::endl;
}

void sift::log_features(const std::vector<std::vector<cv::KeyPoint>>& keypoints) {
    for (size_t i = 0; i < keypoints.size(); ++i) {
        std::cout << "image " << i + 1 << ": " << keypoints[i].size() << " keypoints detected" << std::endl;
    }
}

std::vector<cv::Mat> sift::load_images(const std::vector<std::string>& paths) {
    std::vector<cv::Mat> images;
    for (const auto& path : paths) {
        cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "could not open or find the image: " << path << std::endl;
            continue;
        }
        images.push_back(image);
    }
    return images;
}