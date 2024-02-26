#ifndef UTILS_H
#define UTILS_H

#include "image_handler.h"

#include <vector>
#include <algorithm>
#include <numeric>

// Function to calculate the percentage similarity between two images
void get_percentage(const Image& image1, const Image& image2, double& percentage);

// Function to calculate weights between two images
void get_weights(const Image& image1, const Image& image2, std::vector<double>& weights);

// Function to make prediction based on image comparison
void get_prediction(const Image& image1, const Image& image2, double threshold);

#endif // UTILS_H
