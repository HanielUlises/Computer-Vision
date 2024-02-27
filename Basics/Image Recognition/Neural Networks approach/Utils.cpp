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

Matrix transpose (Matrix a){
    Matrix result (a.n, a.m);
    int i, j;

     for(i = 0; i < a.n; i++){
        for(j = 0; j < a.m; j++){
            a[j][i] = a[i][j];
        }
    }

    return result;
}

Matrix multiply (Matrix a, Matrix b){
    Matrix result (a.n, a.m);

    int i,j,k;
    
    for(i = 0; i < a.n; i++){
        for(j = 0; j < b.m; j++){
            for(k = 0; k < a.m; k++){
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}

double sigmoid (double x){
    return 1.0/(1.0 + std::exp(-x));
}

double sigmoid_dx (double x){
    return x * (1.0 - x);
}

Matrix sigmoid_mat (Matrix a){
    Matrix result (a.n, a.m);
    int i, j;

    for(i = 0; i < a.n; i++){
        for(j = 0; j < a.m; j++){
            result[i][j] = sigmoid(a[i][j]);
        }
    }
    return result;
}

Matrix sigmoid_dx_mat (Matrix a){
    Matrix result (a.n, a.m);
    int i, j;

    for(i = 0; i < a.n; i++){
        for(j = 0; j < a.m; j++){
            result[i][j] = sigmoid_dx (a[i][j]);
        }
    }
    return result;
}

Neural_Network::Neural_Network () {}

Neural_Network::Neural_Network (std::vector<int> sz, double alpha){
    int i;

    n = (int)(sz.size());
    size = sz;

    w.resize(n-1);
    b.resize(n-1);

    nabla_w.resize(n-1);
    nabla_b.resize(n-1);

    for(i = 0; i < n; i++){
        w[i] = Matrix(size[i], size[i+1]);
        b[i] = Matrix(1, size[i + 1]);

        nabla_b[i] = Matrix (size[i], size[i+1]);
        nabla_w[i] = Matrix(1, size[i + 1]);
    }

    learning_rate = alpha;
}

Matrix Neural_Network::feed_forward (Matrix input){
    int i;

    Matrix output(input.n, input.m);

    for(i = 0; i < n - 1; i++){
        output = sigmoid_mat(add(multiply(input, w[i]), b[i]));
    }

    return output;
}

void Neural_Network::backpropagation(Matrix input, Matrix output){
    std::vector<Matrix> l;
    Matrix delta;

    int i;

    l.push_back(input);
    for(i = 0; i < n - 1; i++){
        input = sigmoid_mat(add(multiply(input, w[i]), b[i]));
        l.push_back(input);
    }

    delta = term_by_term(substract(input, output), sigmoid_dx_mat(l[n-1]));

    nabla_b[n - 2].add(delta);
    nabla_w[n - 2].add(multiply(transpose(l[n-1]), delta));

    for(i = n - 3; i >= 0; i--){
        delta = multiply(delta, transpose(w[i+1]));

        delta = term_by_term(delta, sigmoid_dx_mat(l[i+1]));

        nabla_b[i].add(delta);
        nabla_w[i].add(multiply(transpose(l[i]), delta));
    }
}

void Neural_Network::train (std::vector<Matrix> inputs, std::vector<Matrix> outputs){
    int i, j, k;

    for(i = 0; i < n; i++){
        nabla_w[i].zero();
        nabla_b[i].zero();
    }

    for(i = 0; i <(int)(inputs.size()); i++){
        backpropagation(inputs[i], outputs[i]);
    }

    for(i = 0; i < n - 1; i++){
        for(j = 0; j < nabla_w[i].n; j++){
            for(k = 0; k < nabla_w[i].m; k++){
                nabla_w[i][j][k] /= (double)(inputs.size());
                w[i][j][k] -= learning_rate * nabla_w[i][j][k];
            }
        }

        for(j = 0; j < nabla_b[i].n; i++){
            for(k = 0; k < nabla_b[i].m; k++){
                nabla_b[i][j][k] /= (double)(inputs.size());
                b[i][j][k] -= learning_rate * nabla_b[i][j][k];
            }
        }
    }
}