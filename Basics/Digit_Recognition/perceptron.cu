#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"

// Softmax activation function implemented in device code
__device__ void softmax(double* input, double* output, int length) {
    double max_input = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_input) {
            max_input = input[i];
        }
    }

    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        output[i] = exp(input[i] - max_input);
        sum += output[i];
    }

    for (int i = 0; i < length; i++) {
        output[i] /= sum;
    }
}

// CUDA kernel to train the perceptron
__global__ void train_perceptron_kernel(double *weights, double *biases, double *inputs, double *labels, int input_size, int num_classes, int num_samples, double learning_rate) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample < num_samples) {
        double *input = &inputs[sample * input_size];
        double *label = &labels[sample * num_classes];

        double weighted_sum[num_classes];
        double output[num_classes];

        for (int j = 0; j < num_classes; j++) {
            weighted_sum[j] = 0.0;
            for (int i = 0; i < input_size; i++) {
                weighted_sum[j] += input[i] * weights[i * num_classes + j];
            }
            weighted_sum[j] += biases[j];
        }

        softmax(weighted_sum, output, num_classes);

        for (int j = 0; j < num_classes; j++) {
            double error = label[j] - output[j];
            for (int i = 0; i < input_size; i++) {
                weights[i * num_classes + j] += learning_rate * error * input[i];
            }
            biases[j] += learning_rate * error;
        }
    }
}

// Host function to train perceptron using CUDA
void train_perceptron_cuda(Perceptron* p, double **inputs, double **labels, int num_samples, int epochs, double learning_rate) {
    double *d_weights, *d_biases, *d_inputs, *d_labels;

    size_t weights_size = p->input_size * p->num_classes * sizeof(double);
    size_t biases_size = p->num_classes * sizeof(double);
    size_t inputs_size = num_samples * p->input_size * sizeof(double);
    size_t labels_size = num_samples * p->num_classes * sizeof(double);

    cudaMalloc(&d_weights, weights_switch);
    cudaMalloc(&d_biases, biases_size);
    cudaMalloc(&d_inputs, inputs_size);
    cudaMalloc(&d_labels, labels_size);

    cudaMemcpy(d_weights, p->weights, weights_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, p->biases, biases_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputs, *inputs, inputs_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, *labels, labels_size, cudaMemcpyHostToDevice);

    int block_size = 256; // Adjust based on your GPU capabilities (mine is Nvidia RTX 3060 Mobile/Max-Q) 
    int num_blocks = (num_samples + block_size - 1) / block_size;

    for (int epoch = 0; epoch < epochs; epoch++) {
        train_perceptron_kernel<<<num_blocks, block_size>>>(d_weights, d_biases, d_inputs, d_labels, p->input_size, p->num_classes, num_samples, learning_rate);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(p->weights, d_weights, weights_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(p->biases, d_biases, biases_size, cudaMemcpyDeviceToHost);

    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_inputs);
    cudaFree(d_labels);
}

