#include "perceptron.h"
#include <omp.h>

/**
// Sigmoid activation function
static double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
static double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

*/

// Softmax activation function
static void softmax(double* input, double* output, int length) {
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

Perceptron* create_perceptron(int input_size, int num_classes) {
    Perceptron* p = (Perceptron*)malloc(sizeof(Perceptron));
    p->input_size = input_size;
    p->num_classes = num_classes;
    p->weights = (double*)malloc(input_size * num_classes * sizeof(double));
    p->biases = (double*)malloc(num_classes * sizeof(double));

    // Weights and biases with random values
    for (int i = 0; i < input_size * num_classes; i++) {
        p->weights[i] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0; // Random values between -1 and 1
    }
    for (int i = 0; i < num_classes; i++) {
        p->biases[i] = ((double)rand() / (double)RAND_MAX) * 2.0 - 1.0; // Random values between -1 and 1
    }
    return p;
}

// Train the perceptron
void train_perceptron(Perceptron* p, double **inputs, double **labels, int num_samples, int epochs, double learning_rate) {
    #pragma omp parallel for
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int sample = 0; sample < num_samples; sample++) {
            double* input = inputs[sample];
            double* label = labels[sample];

            // Forward pass
            double weighted_sum[p->num_classes];
            double output[p->num_classes];
            for (int j = 0; j < p->num_classes; j++) {
                weighted_sum[j] = 0.0;
                for (int i = 0; i < p->input_size; i++) {
                    weighted_sum[j] += input[i] * p->weights[i * p->num_classes + j];
                }
                weighted_sum[j] += p->biases[j];
            }

            softmax(weighted_sum, output, p->num_classes);

            // Backward pass and weight update
            for (int j = 0; j < p->num_classes; j++) {
                double error = label[j] - output[j];
                for (int i = 0; i < p->input_size; i++) {
                    p->weights[i * p->num_classes + j] += learning_rate * error * input[i];
                }
                p->biases[j] += learning_rate * error;
            }
        }
    }
}

// Predict using the perceptron
int predict_perceptron(Perceptron* p, double *input) {
    double weighted_sum[p->num_classes];
    double output[p->num_classes];
    for (int j = 0; j < p->num_classes; j++) {
        weighted_sum[j] = 0.0;
        for (int i = 0; i < p->input_size; i++) {
            weighted_sum[j] += input[i] * p->weights[i * p->num_classes + j];
        }
        weighted_sum[j] += p->biases[j];
    }

    softmax(weighted_sum, output, p->num_classes);

    int predicted_class = 0;
    double max_output = output[0];
    for (int j = 1; j < p->num_classes; j++) {
        if (output[j] > max_output) {
            max_output = output[j];
            predicted_class = j;
        }
    }
    return predicted_class;
}

void free_perceptron(Perceptron* p) {
    free(p->weights);
    free(p->biases);
    free(p);
}

// Save the perceptron to a file
void save_perceptron(Perceptron* p, const char* filename) {
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Failed to open file for saving perceptron");
        return;
    }
    fwrite(&p->input_size, sizeof(int), 1, file);
    fwrite(&p->num_classes, sizeof(int), 1, file);
    fwrite(p->weights, sizeof(double), p->input_size * p->num_classes, file);
    fwrite(p->biases, sizeof(double), p->num_classes, file);
    fclose(file);
}

// Loads the perceptron from a file
void load_perceptron(Perceptron* p, const char* filename) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        perror("Failed to open file for loading perceptron");
        return;
    }
    fread(&p->input_size, sizeof(int), 1, file);
    fread(&p->num_classes, sizeof(int), 1, file);
    fread(p->weights, sizeof(double), p->input_size * p->num_classes, file);
    fread(p->biases, sizeof(double), p->num_classes, file);
    fclose(file);
}
