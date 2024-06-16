#ifndef UTILS_H
#define UTILS_H

#include "image_handler.h"
#include <vector>

void get_percentage(const Image& image1, const Image& image2, double& percentage);
void get_weights(const Image& image1, const Image& image2, std::vector<double>& weights);
void get_prediction_percentage(const Image& image1, const Image& image2, double threshold);
void get_prediction_exhaustive(const Image& image1, const Image& image2, double threshold);

#endif // UTILS_H
