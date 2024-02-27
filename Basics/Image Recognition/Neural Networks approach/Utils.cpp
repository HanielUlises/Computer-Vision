#include "Utils.h"

xorShift::xorShift (){
    x = 346535763;
}

unsigned xorShift::next (){
    unsigned t = x ^ (x << 11);

    x = y;
    y = z;
    z = w;

    return w = w ^ (w >> 19) ^ t ^ (t >> 8);
}

Matrix::Matrix (){
    n = 0;
    m = 0;
    a = nullptr;

}

Matrix::Matrix (int rows, int columns){
    int i, j;

    n = rows;
    m = columns;

    a = new double* [n];
    for(i = 0; i < n; i++){
        a[i] = new double [m];

        for(j = 0; j < m; j++){
            a[i][j] = 0.0;
        }
    }
} 

Matrix::Matrix (const Matrix& x){
    int i, j;

    n = x.n;
    m = x.m;

    a = new double* [n];

    for(i = 0; i < n; i++){
        a[i] = new double [m];

        for(j = 0; j < m; j++){
            a[i][j] = x.a[i][j];
        }
    }
}

void Matrix::operator=(const Matrix& x){
    int i, j;

    n = x.n;
    m = x.m;

    a = new double* [n];

    for(i = 0; i < n; i++){
        a[i] = new double [m];

        for(j = 0; j < m; j++){
            a[i][j] = x.a[i][j];
        }
    }
}

void Matrix::randomize (){
    int i, j;

    for(i = 0; i < n; i++){
        for(j = 0; j < m; j++){
            a[i][j] = rand_gen();

            if(random_number_gen.next() % 2 == 0){
                a[i][j] = -a[i][j];
            }
        }
    }
}

void Matrix::zero (){
    int i, j;

    for(i = 0; i < n; i++){
        for(j = 0; j < m; j++){
            a[i][j] = 0.0f;
        }
    }
}

void Matrix::add (Matrix x){
    int i, j;

    for(i = 0; i < n; i++){
        for(j = 0; j < m; j++){
            a[i][j] += x[i][j];
        }
    }
}

double* &Matrix::operator [](const int& idx){
    return a[idx];
}

double rand_gen (){
    return random_number_gen.next() % 1000000001 / 1000000000.0;
}

Matrix add (Matrix a, Matrix b){
    Matrix result (a.n, a.m);
    int i, j;

    for(i = 0; i < a.n; i++){
        for(j = 0; j < a.m; j++){
            result[i][j] = a[i][j] + b[i][j];
        }
    }

    return result;
}

Matrix substract (Matrix a, Matrix b){
    Matrix result (a.n, a.m);
    int i, j;

    for(i = 0; i < a.n; i++){
        for(j = 0; j < a.m; j++){
            result[i][j] = a[i][j] - b[i][j];
        }
    }

    return result;
}