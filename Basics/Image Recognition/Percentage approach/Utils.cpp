#include "Utils.h"
#include <iostream>
#include <cmath> // For std::abs

// Function to calculate the percentage similarity between two images
double calculate_percentage_similarity(const Image& image1, const Image& image2) {
    auto image_matrix1 = image1.get_matrix();
    auto image_matrix2 = image2.get_matrix();

    // Ensure both images have the same dimensions (for simplicity)
    if (image_matrix1.size() != image_matrix2.size() || image_matrix1[0].size() != image_matrix2[0].size()) {
        std::cerr << "Images must have the same dimensions for comparison." << std::endl;
        return 0.0;
    }

    int total_pixels = 0;
    int matching_pixels = 0;

    for (size_t i = 0; i < image_matrix1.size(); ++i) {
        for (size_t j = 0; j < image_matrix1[i].size(); ++j) {
            if (image_matrix1[i][j] == image_matrix2[i][j]) {
                matching_pixels++;
            }
            total_pixels++;
        }
    }

    return total_pixels > 0 ? static_cast<double>(matching_pixels) / total_pixels * 100.0 : 0.0;
}

// Function to predict how similar two images are
void predict_image_similarity(const Image& image1, const Image& image2) {
    double similarity_percentage = calculate_percentage_similarity(image1, image2);

    std::cout << "Similarity Percentage: " << similarity_percentage << "%" << std::endl;

    // Determine the similarity level based on the percentage
    if (similarity_percentage > 90.0) {
        std::cout << "The images are highly similar." << std::endl;
    } else if (similarity_percentage > 70.0) {
        std::cout << "The images are moderately similar." << std::endl;
    } else if (similarity_percentage > 50.0) {
        std::cout << "The images have some similarity." << std::endl;
    } else {
        std::cout << "The images are not very similar." << std::endl;
    }
}

std::vector<double> computeMSEGradient(const std::vector<double>& predictions, const std::vector<double>& targets) {
    std::vector<double> gradients(predictions.size());
    for (size_t i = 0; i < predictions.size(); ++i) {
        gradients[i] = 2.0 * (predictions[i] - targets[i]);
    }
    return gradients;
}

void softmax(std::vector<double>& outputs) {
    double max = *max_element(outputs.begin(), outputs.end());
    double sum = 0.0;

    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i] = exp(outputs[i] - max); // Improve numerical stability
        sum += outputs[i];
    }

    for (size_t i = 0; i < outputs.size(); ++i) {
        outputs[i] /= sum;
    }
}