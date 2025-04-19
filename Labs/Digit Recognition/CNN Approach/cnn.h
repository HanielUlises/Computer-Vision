#ifndef CNN_H
#define CNN_H

#include <stdint.h>

// Structure for a convolutional layer
typedef struct {
    int num_filters;
    int filter_size;
    int stride;
    int input_depth;
    int input_size;
    int output_size;
    float *filters;
    float *biases;
    float *outputs;
    float *d_filters;
    float *d_biases;
    float *d_outputs;
} ConvLayer;

// Structure for a dense (fully connected) layer
typedef struct {
    int input_size;
    int output_size;
    float *weights;
    float *biases;
    float *outputs;
    float *d_weights;
    float *d_biases;
    float *d_outputs;
} DenseLayer;

void conv_forward(ConvLayer *layer, float *input);
void conv_backward(ConvLayer *layer, float *input, float *d_input);

void dense_forward(DenseLayer *layer, float *input);
void dense_backward(DenseLayer *layer, float *input, float *d_input);

void relu_forward(float *input, int size);
void relu_backward(float *input, float *d_input, int size);

void softmax_forward(float *input, int size);
void softmax_backward(float *input, float *d_input, int size);

void cnn_train(ConvLayer *conv, DenseLayer *dense, float *input, float *labels, int num_epochs, float learning_rate);
float cnn_predict(ConvLayer *conv, DenseLayer *dense, float *input);

#endif // CNN_H
