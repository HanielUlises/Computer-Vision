#include "HOGExtractor.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cmath>

HOGExtractor::HOGExtractor(cv::Size winSize, cv::Size blockSize, cv::Size blockStride, 
                           cv::Size cellSize, int nbins)
    : winSize(winSize), blockSize(blockSize), blockStride(blockStride), 
      cellSize(cellSize), nbins(nbins) {
    hog = cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);
}

bool HOGExtractor::loadImage(const std::string& imagePath) {
    inputImage = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load image from " << imagePath << std::endl;
        return false;
    }

    if (inputImage.size() != winSize) {
        cv::resize(inputImage, inputImage, winSize, 0, 0, cv::INTER_LINEAR);
    }
    
    return true;
}

bool HOGExtractor::computeHOG() {
    if (inputImage.empty()) {
        std::cerr << "Error: No image loaded for HOG computation" << std::endl;
        return false;
    }

    descriptors.clear();

    try {
        hog.compute(inputImage, descriptors);
        std::cout << "HOG descriptors computed. Size: " << descriptors.size() << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error computing HOG descriptors: " << e.what() << std::endl;
        return false;
    }
}

std::vector<float> HOGExtractor::getDescriptors() const {
    return descriptors;
}

// Visualize HOG descriptors
cv::Mat HOGExtractor::visualizeHOG() const {
    if (descriptors.empty()) {
        std::cerr << "Error: No HOG descriptors to visualize" << std::endl;
        return cv::Mat();
    }

    cv::Mat colorImage;
    cv::cvtColor(inputImage, colorImage, cv::COLOR_GRAY2BGR);

    float scale = 10.0f; 
    int cellPixels = cellSize.width; 
    int cellsPerBlock = blockSize.width / cellSize.width;
    int binsPerCell = nbins;

    // Calculate number of cells in x and y directions
    int cellsX = winSize.width / cellSize.width;
    int cellsY = winSize.height / cellSize.height;

    // Compute descriptor dimensions
    int blockStepX = blockStride.width / cellSize.width;
    int blockStepY = blockStride.height / cellSize.height;
    int blocksX = (cellsX - cellsPerBlock) / blockStepX + 1;
    int blocksY = (cellsY - cellsPerBlock) / blockStepY + 1;

    // For each block
    int descriptorIndex = 0;
    for (int by = 0; by < blocksY; ++by) {
        for (int bx = 0; bx < blocksX; ++bx) {
            // Cells for each block
            for (int cy = 0; cy < cellsPerBlock; ++cy) {
                for (int cx = 0; cx < cellsPerBlock; ++cx) {
                    // Cell position
                    int cellX = bx * blockStepX + cx;
                    int cellY = by * blockStepY + cy;

                    // Center of the cell in pixels
                    float centerX = (cellX + 0.5f) * cellPixels;
                    float centerY = (cellY + 0.5f) * cellPixels;

                    // Histogram for a given cell
                    for (int bin = 0; bin < binsPerCell; ++bin) {
                        float magnitude = descriptors[descriptorIndex];
                        descriptorIndex++;

                        // Orientation
                        float angle = bin * (180.0f / binsPerCell);
                        float rad = angle * CV_PI / 180.0f;

                        // Calculate line endpoints
                        float dx = scale * magnitude * cos(rad);
                        float dy = scale * magnitude * sin(rad);
                        cv::Point startPoint(centerX - dx, centerY - dy);
                        cv::Point endPoint(centerX + dx, centerY + dy);

                        // Draw gradient line
                        cv::line(colorImage, startPoint, endPoint, cv::Scalar(0, 255, 0), 1);
                    }
                }
            }
        }
    }

    return colorImage;
}

bool HOGExtractor::saveVisualization(const std::string& outputPath) const {
    cv::Mat visualization = visualizeHOG();
    if (visualization.empty()) {
        std::cerr << "Error: No visualization to save" << std::endl;
        return false;
    }

    try {
        cv::imwrite(outputPath, visualization);
        std::cout << "HOG visualization saved to " << outputPath << std::endl;
        return true;
    } catch (const cv::Exception& e) {
        std::cerr << "Error saving visualization: " << e.what() << std::endl;
        return false;
    }
}