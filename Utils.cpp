#include "image_handler.h"
#include "Utils.h"

#include <iostream>

// Function to calculate the percentage similarity between two images
void get_percentage(const Image& image1, const Image& image2, double& percentage) {
    std::vector<std::vector<char>> image_matrix1 = image1.get_matrix();
    std::vector<std::vector<char>> image_matrix2 = image2.get_matrix();

    // Assuming both images have the same dimensions
    int total_pixels = 0;
    int matching_pixels = 0;

    for (size_t i = 0; i < image_matrix1.size(); ++i) {
        for (size_t j = 0; j < image_matrix2[i].size(); ++j) {
            // Compare pixel values and count matching pixels
            if (image_matrix1[i][j] == image_matrix2[i][j]) {
                matching_pixels++;
            }
            total_pixels++;
        }
    }

    // Calculate percentage similarity
    if (total_pixels > 0) {
        percentage = static_cast<double>(matching_pixels) / total_pixels * 100.0;
    } else {
        percentage = 0.0;
    }
    std::cout << "Percentage of similarity: " << percentage << std::endl;
}

// Function to calculate weights between two images
void get_weights(const Image& image1, const Image& image2, std::vector<double>& weights) {
    std::vector<std::vector<char>> image_matrix1 = image1.get_matrix();
    std::vector<std::vector<char>> image_matrix2 = image2.get_matrix();

    // Both images must have the same dimensions
    for (size_t i = 0; i < image_matrix1.size(); ++i) {
        for (size_t j = 0; j < image_matrix1[i].size(); ++j) {
            weights.push_back(std::abs(image_matrix1[i][j] - image_matrix2[i][j]));
        }
    }
}

void get_prediction(const Image& image1, const Image& image2, double threshold) {
    double percentage;
    get_percentage(image1, image2, percentage);

    std::vector<double> weights;
    get_weights(image1, image2, weights);

    if (percentage >= threshold) {
        std::cout << "Images are similar based on percentage similarity." << std::endl;
    } else {
        double totalWeight = std::accumulate(weights.begin(), weights.end(), 0.0);
        if (totalWeight <= threshold) {
            std::cout << "Images are similar based on total weight." << std::endl;
        } else {
            std::cout << "Images are not similar." << std::endl;
        }
    }
}
