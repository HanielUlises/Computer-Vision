#ifndef UTILS_H
#define UTILS_H

#include <numeric>
#include <cmath>
#include <vector>
#include <cassert>
#include <string>
#include <fstream>
#include <iostream>

//Shift-register generator
struct xorShift{
    xorShift();

    unsigned x,y,z,w;
    unsigned next ();
};

extern xorShift random_number_gen;
struct Matrix{
    int n, m;
    double **a;

    Matrix ();
    Matrix (int rows, int columns);
    Matrix (const Matrix& m);
    ~Matrix ();

    void operator =(const Matrix& x);
    double* &operator [](const int& idx);

    void randomize ();
    void zero ();
    // Both matrices must be the same dimensions
    void add (Matrix x);
};
struct Neural_Network{
    int n;

    double learning_rate;

    std::vector<int> size;
    // When using nabla (∇) i refer to the gradient
    // w stands for weight matrix
    // b stands for bais matrix
    std::vector<Matrix> w, b, nabla_w, nabla_b;

    Neural_Network ();
    Neural_Network (std::vector<int> sz, double alpha);

    Matrix feed_forward (Matrix input);

    void backpropagation (Matrix input, Matrix output);
    void train (std::vector<Matrix> inputs, std::vector<Matrix> outputs);
};

extern std::vector<Matrix> train_input, train_output;

double rand_gen ();
// Basic arithmetical functions
Matrix add (Matrix a, Matrix b);
Matrix substract (Matrix a, Matrix b);
// Matrix tranpose
Matrix transpose (Matrix a);
// Hadamard product for matrix multiplication
// It is a term by term matrix multplication
Matrix hadamard (Matrix a, Matrix b);
// The matrix product
Matrix multiply (Matrix a, Matrix b);
// σ(x) Sigmoid function
double sigmoid (double x);
// σ'(x) Sigmoid derivative
double sigmoid_dx (double x);

Matrix sigmoid_mat (Matrix a);
Matrix sigmoid_dx_mat (Matrix a);

void time_stamp ();

// Neural network feeding
void parse_training_data ();

std::vector<int> split(std::string x);

#endif // UTILS_H