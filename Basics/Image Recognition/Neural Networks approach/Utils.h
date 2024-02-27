#ifndef UTILS_H
#define UTILS_H

#include <numeric>
#include <cmath>
#include <vector>
struct xorShift{
    xorShift();

    unsigned x,y,z,w;
    unsigned next ();
} random_number_gen ;

struct Matrix{
    int n, m;
    double **a;

    Matrix ();
    Matrix (int rows, int columns);
    Matrix (const Matrix& m);

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
    std::vector<Matrix> w, b, nabla_w, nabla_b;

    Neural_Network ();
    Neural_Network (std::vector<int> sz, double alpha);

    Matrix feed_forward (Matrix input);

    void backpropagation (Matrix input, Matrix output);
    void train (std::vector<Matrix> inputs, std::vector<Matrix> outputs);
};

double rand_gen ();
Matrix add (Matrix a, Matrix b);
Matrix substract (Matrix a, Matrix b);
Matrix transpose (Matrix a);
Matrix multiply (Matrix a, Matrix b);
// σ(x) Sigmoid function
double sigmoid (double x);
// σ'(x) Sigmoid derivative
double sigmoid_dx (double x);

Matrix sigmoid_mat (Matrix a);
Matrix sigmoid_dx_mat (Matrix a);


#endif // UTILS_H