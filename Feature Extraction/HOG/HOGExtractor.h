// HOGExtractor.h
#ifndef HOG_EXTRACTOR_H
#define HOG_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

class HOGExtractor {
private:
    cv::HOGDescriptor hog;
    
    // Parameters for HOG computation
    cv::Size winSize;       // Window size for HOG computation
    cv::Size blockSize;     // Size of blocks
    cv::Size blockStride;   // Stride between blocks
    cv::Size cellSize;      // Size of cells
    int nbins;              // Number of orientation bins
    
    cv::Mat inputImage;
    std::vector<float> descriptors;
    
public:
    HOGExtractor(
        cv::Size winSize = cv::Size(64, 128),
        cv::Size blockSize = cv::Size(16, 16),
        cv::Size blockStride = cv::Size(8, 8),
        cv::Size cellSize = cv::Size(8, 8),
        int nbins = 9
    );

    bool loadImage(const std::string& imagePath);
    bool computeHOG();
    std::vector<float> getDescriptors() const;
    cv::Mat visualizeHOG() const;
    bool saveVisualization(const std::string& outputPath) const;
};

#endif // HOG_EXTRACTOR_H