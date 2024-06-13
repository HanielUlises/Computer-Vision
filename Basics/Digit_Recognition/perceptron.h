#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

typedef struct {
    int input_size;
    int num_classes;
    double *weights;
    double *biases;
} Perceptron;

Perceptron* create_perceptron(int input_size, int num_classes);
void train_perceptron(Perceptron* p, double **inputs, double **labels, int num_samples, int epochs, double learning_rate);
int predict_perceptron(Perceptron* p, double *input);
void free_perceptron(Perceptron* p);
void save_perceptron(Perceptron* p, const char* filename);
void load_perceptron(Perceptron* p, const char* filename);

#ifdef __cplusplus
extern "C" {
#endif

void train_perceptron_cuda(Perceptron* p, double **inputs, double **labels, int num_samples, int epochs, double learning_rate);

#ifdef __cplusplus
}
#endif

#endif // PERCEPTRON_H
