#include "utils.h"
#include <stdlib.h>
#include <string.h>

// Compute gradients for convolutional filters and biases
void compute_conv_gradients(ConvLayer *layer, float *input, float *d_input) {
    int input_size = layer->input_size;
    int filter_size = layer->filter_size;
    int output_size = layer->output_size;
    int num_filters = layer->num_filters;
    int stride = layer->stride;

    memset(layer->d_filters, 0, layer->num_filters * layer->filter_size * layer->filter_size * sizeof(float));
    memset(layer->d_biases, 0, layer->num_filters * sizeof(float));

    for (int f = 0; f < num_filters; f++) {
        for (int y = 0; y < output_size; y++) {
            for (int x = 0; x < output_size; x++) {
                float grad = layer->d_outputs[f * output_size * output_size + y * output_size + x];
                for (int ky = 0; ky < filter_size; ky++) {
                    for (int kx = 0; kx < filter_size; kx++) {
                        int in_y = y * stride + ky;
                        int in_x = x * stride + kx;
                        layer->d_filters[f * filter_size * filter_size + ky * filter_size + kx] +=
                            grad * input[in_y * input_size + in_x];
                    }
                }
                layer->d_biases[f] += grad;
            }
        }
    }
}

// Compute gradients for dense weights and biases
void compute_dense_gradients(DenseLayer *layer, float *input, float *d_input) {
    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            layer->d_weights[i * layer->input_size + j] = d_input[i] * input[j];
        }
        layer->d_biases[i] = d_input[i];
    }
}
