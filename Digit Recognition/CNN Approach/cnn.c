#include "cnn.h"
#include "utils.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Apply a filter to the input image to produce an output feature map
static void conv_apply_filter(ConvLayer *layer, float *input, float *output, int filter_index) {
    int input_size = layer->input_size;   
    int filter_size = layer->filter_size; 
    int stride = layer->stride;           
    int output_size = layer->output_size; 

    memset(output, 0, output_size * output_size * sizeof(float));

    // Perform convolution operation for the given filter
    for (int y = 0; y < output_size; y++) {
        for (int x = 0; x < output_size; x++) {
            float sum = 0.0f;
            for (int ky = 0; ky < filter_size; ky++) {
                for (int kx = 0; kx < filter_size; kx++) {
                    // Compute the position in the input image
                    int in_y = y * stride + ky;
                    int in_x = x * stride + kx;
                    // Convolution sum
                    sum += input[in_y * input_size + in_x] * layer->filters[filter_index * filter_size * filter_size + ky * filter_size + kx];
                }
            }
            // Add bias term and store the result in the output feature map
            output[y * output_size + x] = sum + layer->biases[filter_index];
        }
    }
}

// Perform the forward pass through the convolutional layer
void conv_forward(ConvLayer *layer, float *input) {
    int num_filters = layer->num_filters;
    int output_size = layer->output_size;

    // Apply each filter to the input and store the results in the output feature maps
    for (int f = 0; f < num_filters; f++) {
        conv_apply_filter(layer, input, layer->outputs + f * output_size * output_size, f);
    }
}

// Perform the backward pass to compute gradients for the convolutional layer
void conv_backward(ConvLayer *layer, float *input, float *d_input) {
    int num_filters = layer->num_filters; 
    int filter_size = layer->filter_size; 
    int output_size = layer->output_size; 
    int stride = layer->stride;           

    // Initialize gradients of filters and biases to zero
    memset(layer->d_filters, 0, num_filters * filter_size * filter_size * sizeof(float));
    memset(layer->d_biases, 0, num_filters * sizeof(float));

    // Compute gradients for each filter and bias term
    for (int f = 0; f < num_filters; f++) {
        float *filter_deltas = layer->d_filters + f * filter_size * filter_size;
        float *bias_deltas = layer->d_biases + f;

        for (int y = 0; y < output_size; y++) {
            for (int x = 0; x < output_size; x++) {
                float grad = layer->d_outputs[f * output_size * output_size + y * output_size + x];
                for (int ky = 0; ky < filter_size; ky++) {
                    for (int kx = 0; kx < filter_size; kx++) {
                        int in_y = y * stride + ky;
                        int in_x = x * stride + kx;
                        // Compute the gradient for each filter
                        filter_deltas[ky * filter_size + kx] += grad * input[in_y * layer->input_size + in_x];
                    }
                }
                *bias_deltas += grad;
            }
        }
    }

    // Compute gradients for the input based on the gradients of the output
    for (int y = 0; y < output_size; y++) {
        for (int x = 0; x < output_size; x++) {
            float grad = 0;
            for (int f = 0; f < num_filters; f++) {
                grad += layer->d_outputs[f * output_size * output_size + y * output_size + x] *
                        layer->filters[f * filter_size * filter_size + (y % filter_size) * filter_size + (x % filter_size)];
            }
            d_input[y * layer->input_size + x] = grad;
        }
    }
}

// Forward pass through a dense (fully connected) layer
void dense_forward(DenseLayer *layer, float *input) {
    int input_size = layer->input_size;   
    int output_size = layer->output_size; 

    // Compute the output values for the dense layer
    for (int i = 0; i < output_size; i++) {
        layer->outputs[i] = layer->biases[i]; 
        for (int j = 0; j < input_size; j++) {
            // Sum the weighted inputs
            layer->outputs[i] += layer->weights[i * input_size + j] * input[j];
        }
    }
}

// Backward pass to compute gradients for the dense (fully connected) layer
void dense_backward(DenseLayer *layer, float *input, float *d_input) {
    // Compute gradients for weights, biases, and inputs
    compute_dense_gradients(layer, input, d_input);
}

// ReLU activation function: outputs the input if it is positive, else zero
void relu_forward(float *input, int size) {
    for (int i = 0; i < size; i++) {
        if (input[i] < 0) input[i] = 0;
    }
}

// Gradient of ReLU activation function: gradient is 1 if input is positive, else 0
void relu_backward(float *input, float *d_input, int size) {
    for (int i = 0; i < size; i++) {
        d_input[i] = (input[i] > 0) ? d_input[i] : 0;
    }
}

// Softmax activation function: converts logits to probabilities
void softmax_forward(float *input, int size) {
    float max = input[0];
    // Find the maximum value in the input for numerical stability
    for (int i = 1; i < size; i++) {
        if (input[i] > max) max = input[i];
    }

    float sum = 0;
    // Compute the exponentials of the input values
    for (int i = 0; i < size; i++) {
        input[i] = exp(input[i] - max);
        sum += input[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

// Backward pass for softmax activation function: computes gradients with respect to input
void softmax_backward(float *input, float *d_input, int size) {
    // Allocate memory for the Jacobian matrix of the softmax function
    float *jacobian = (float *)malloc(size * size * sizeof(float));
    softmax_jacobian(input, jacobian, size);

    // Compute the gradient of the loss with respect to the softmax inputs
    for (int i = 0; i < size; i++) {
        float grad = 0;
        for (int j = 0; j < size; j++) {
            grad += jacobian[i * size + j] * d_input[j];
        }
        d_input[i] = grad;
    }

    free(jacobian);
}

// Compute cross-entropy loss between predictions and true labels
float cross_entropy_loss(float *predictions, uint8_t *labels, int num_samples) {
    float loss = 0.0f;
    // Accumulate loss for each sample
    for (int i = 0; i < num_samples; i++) {
        loss -= log(predictions[i * 10 + labels[i]]);  
    }
    // Average loss over all samples
    return loss / num_samples;
}

// Compute the gradient of the cross-entropy loss with respect to predictions
void cross_entropy_loss_gradient(float *predictions, uint8_t *labels, float *gradients, int num_samples) {
    // Initialize gradients to zero
    memset(gradients, 0, num_samples * 10 * sizeof(float));  
    // Update gradients based on the true labels
    for (int i = 0; i < num_samples; i++) {
        gradients[i * 10 + labels[i]] -= 1.0f / num_samples;
    }
}

// Train the CNN using a simple gradient descent algorithm
void cnn_train(ConvLayer *conv, DenseLayer *dense, float *input, uint8_t *labels, int num_samples, int num_epochs, float learning_rate) {
    int num_classes = 10;  
    float *predictions = (float *)malloc(num_samples * num_classes * sizeof(float));
    float *d_predictions = (float *)malloc(num_samples * num_classes * sizeof(float));

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass through the network
        conv_forward(conv, input);
        relu_forward(conv->outputs, conv->output_size * conv->output_size * conv->num_filters);
        dense_forward(dense, conv->outputs);
        softmax_forward(dense->outputs, num_classes);

        // Compute loss and gradients
        float loss = cross_entropy_loss(dense->outputs, labels, num_samples);
        printf("Epoch %d: Loss = %f\n", epoch, loss);
        
        cross_entropy_loss_gradient(dense->outputs, labels, d_predictions, num_samples);
        softmax_backward(dense->outputs, d_predictions, num_classes);
        dense_backward(dense, conv->outputs, d_predictions);
        conv_backward(conv, input, d_predictions);

        // Update weights and biases using gradient descent
        for (int i = 0; i < conv->num_filters * conv->filter_size * conv->filter_size; i++) {
            conv->filters[i] -= learning_rate * conv->d_filters[i];
        }
        for (int i = 0; i < conv->num_filters; i++) {
            conv->biases[i] -= learning_rate * conv->d_biases[i];
        }
        for (int i = 0; i < dense->input_size * dense->output_size; i++) {
            dense->weights[i] -= learning_rate * dense->d_weights[i];
        }
        for (int i = 0; i < dense->output_size; i++) {
            dense->biases[i] -= learning_rate * dense->d_biases[i];
        }
    }

    free(predictions);
    free(d_predictions);
}

// Predict the class for a given input using the trained CNN
float cnn_predict(ConvLayer *conv, DenseLayer *dense, float *input) {
    int num_classes = 10; 
    float *outputs = (float *)malloc(num_classes * sizeof(float));

    // Forward pass through the network
    conv_forward(conv, input);
    relu_forward(conv->outputs, conv->output_size * conv->output_size * conv->num_filters);
    dense_forward(dense, conv->outputs);
    softmax_forward(dense->outputs, num_classes);

    // Determine the class with the highest probability
    float max_prob = -INFINITY;
    int predicted_class = -1;
    for (int i = 0; i < num_classes; i++) {
        if (dense->outputs[i] > max_prob) {
            max_prob = dense->outputs[i];
            predicted_class = i;
        }
    }

    // Free allocated memory
    free(outputs);
    return predicted_class;
}
