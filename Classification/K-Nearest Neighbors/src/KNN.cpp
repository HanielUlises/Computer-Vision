#include "KNN.h"
#include <limits>
#include <algorithm>

KNN::KNN(int k) : k(k) {}

void KNN::train(const std::vector<cv::Mat>& images, const std::vector<int>& labels) {
    trainImages = images;
    trainLabels = labels;
}

int KNN::predict(const cv::Mat& image) {
    std::vector<std::pair<double, int>> distances;
    for (size_t i = 0; i < trainImages.size(); i++) {
        double dist = calculateDistance(trainImages[i], image);
        distances.emplace_back(dist, trainLabels[i]);
    }

    std::sort(distances.begin(), distances.end());
    
    std::vector<int> kVotes(k);
    for (int i = 0; i < k; ++i) {
        kVotes[i] = distances[i].second;
    }

    std::sort(kVotes.begin(), kVotes.end());
    int mostCommon = kVotes[0], maxCount = 1, currentCount = 1;
    for (int i = 1; i < k; ++i) {
        if (kVotes[i] == kVotes[i - 1]) {
            currentCount++;
            if (currentCount > maxCount) {
                maxCount = currentCount;
                mostCommon = kVotes[i];
            }
        } else {
            currentCount = 1;
        }
    }
    return mostCommon;
}

double KNN::calculateDistance(const cv::Mat& a, const cv::Mat& b) {
    cv::Mat diff;
    cv::absdiff(a, b, diff);
    diff = diff.mul(diff);  // Squaring each element
    return cv::sum(diff)[0];  // Summing all elements
}
