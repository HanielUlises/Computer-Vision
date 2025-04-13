// main.cpp
#include "HOGExtractor.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    HOGExtractor hogExtractor(
        cv::Size(64, 128),    // Window size
        cv::Size(16, 16),     // Block size
        cv::Size(8, 8),       // Block stride
        cv::Size(8, 8),       // Cell size
        9                     // Number of bins
    );

    std::string imagePath = argv[1];
    if (!hogExtractor.loadImage(imagePath)) {
        return -1;
    }

    if (!hogExtractor.computeHOG()) {
        return -1;
    }

    std::vector<float> descriptors = hogExtractor.getDescriptors();
    std::cout << "Number of HOG descriptors: " << descriptors.size() << std::endl;

    std::string outputPath = "hog_visualization.jpg";
    if (!hogExtractor.saveVisualization(outputPath)) {
        return -1;
    }

    cv::Mat visualization = hogExtractor.visualizeHOG();
    if (!visualization.empty()) {
        cv::imshow("HOG Visualization", visualization);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    return 0;
}