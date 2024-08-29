#include "SIFT.h"
#include <fstream>
#include <iostream>

SIFT::SIFT() {
    sift = cv::xfeatures2d::SIFT::create();
}

// Extract SIFT features from images and save them to a CSV file
void SIFT::extractToCSV(const std::vector<cv::Mat>& imgs, const std::string& outCSV) {
    std::vector<std::vector<cv::KeyPoint>> allKps;
    std::vector<cv::Mat> allDescs;

    for (const auto& img : imgs) {
        std::vector<cv::KeyPoint> kps;
        cv::Mat descs;

        // Detect keypoints and compute descriptors
        sift->detectAndCompute(img, cv::noArray(), kps, descs);

        allKps.push_back(kps);
        allDescs.push_back(descs);

        std::cout << "Extracted " << kps.size() << " keypoints from an image." << std::endl;
    }

    // Log extracted features
    logFeatures(allKps);

    writeCSV(allKps, allDescs, outCSV);
}

void SIFT::writeCSV(const std::vector<std::vector<cv::KeyPoint>>& kps, const std::vector<cv::Mat>& descs, const std::string& outCSV) {
    std::ofstream csvFile(outCSV);

    if (!csvFile.is_open()) {
        std::cerr << "Could not open the output CSV file!" << std::endl;
        return;
    }

    for (size_t i = 0; i < kps.size(); ++i) {
        for (size_t j = 0; j < kps[i].size(); ++j) {
            const cv::KeyPoint& kp = kps[i][j];
            const cv::Mat& desc = descs[i].row(j);

            // Write keypoint info: x, y, size, angle, response, octave, class_id
            csvFile << kp.pt.x << "," << kp.pt.y << "," << kp.size << "," << kp.angle << ","
                    << kp.response << "," << kp.octave << "," << kp.class_id;

            // Write descriptor values
            for (int k = 0; k < desc.cols; ++k) {
                csvFile << "," << desc.at<float>(k);
            }
            csvFile << "\n";
        }
    }

    csvFile.close();
    std::cout << "Features written to " << outCSV << std::endl;
}

void SIFT::logFeatures(const std::vector<std::vector<cv::KeyPoint>>& kps) {
    for (size_t i = 0; i < kps.size(); ++i) {
        std::cout << "Image " << i + 1 << ": " << kps[i].size() << " keypoints detected." << std::endl;
    }
}

std::vector<cv::Mat> SIFT::loadImages(const std::vector<std::string>& paths) {
    std::vector<cv::Mat> imgs;
    for (const auto& path : paths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Could not open or find the image: " << path << std::endl;
            continue;
        }
        imgs.push_back(img);
    }
    return imgs;
}
