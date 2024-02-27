#ifndef UTILS_H
#define UTILS_H

#include "image_handler.h"

#include <vector>

// Function to calculate the percentage similarity between two images
double calculate_percentage_similarity(const Image& image1, const Image& image2);

// Function to predict how similar two images are
void predict_image_similarity(const Image& image1, const Image& image2);

#endif // UTILS_H