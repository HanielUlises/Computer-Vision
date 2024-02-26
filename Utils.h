#include "image_handler.h"

#include <numeric>
#include <algorithm>

// Function to calculate the percentage similarity between two images
void get_percentage(Image& image1, Image& image2, double& percentage);

// Function to calculate weights between two images
void get_weights(Image& image1, Image& image2, std::vector<double> &weights);

void get_prediction(Image& image1, Image& image2, double threshold);
