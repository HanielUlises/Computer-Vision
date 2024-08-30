#ifndef UTILS_H
#define UTILS_H

#include "cnn.h"
#include <stdint.h>

void compute_conv_gradients(ConvLayer *layer, float *input, float *d_input);
void compute_dense_gradients(DenseLayer *layer, float *input, float *d_input);

#endif // UTILS_H
