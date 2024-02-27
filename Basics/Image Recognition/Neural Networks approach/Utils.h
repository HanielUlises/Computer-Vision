#ifndef UTILS_H
#define UTILS_H

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

double rand_gen ();
Matrix add (Matrix a, Matrix b);
Matrix substract (Matrix a, Matrix b);


#endif // UTILS_H