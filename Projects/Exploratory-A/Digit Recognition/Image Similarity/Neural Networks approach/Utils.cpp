#include "Utils.h"

xorShift random_number_gen;

std::vector<Matrix> train_input, train_output;

xorShift::xorShift (){
    x = 346535763;
    y = 543212235;
    z = 4213323423;
    w = 3423412566;
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
            a[i][j] = 0.0f;
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

Matrix::~Matrix (){
    for(int i = 0; i < n; i++){
        delete[] a[i];
    }
    delete[] a;
}

void Matrix::operator=(const Matrix& x) {
    if (this != &x) {
        for (int i = 0; i < n; i++) {
            delete[] a[i];
        }
        delete[] a;

        n = x.n;
        m = x.m;

        a = new double*[n];
        for (int i = 0; i < n; i++) {
            a[i] = new double[m];
            for (int j = 0; j < m; j++) {
                a[i][j] = x.a[i][j];
            }
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

void Matrix::add(Matrix x) {
    if (n != x.n || m != x.m) return;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            a[i][j] += x.a[i][j];
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

Matrix transpose(Matrix a) {
    Matrix result(a.m, a.n);
    for (int i = 0; i < a.n; i++) {
        for (int j = 0; j < a.m; j++) {
            result[j][i] = a[i][j];
        }
    }
    return result;
}

Matrix hadamard (Matrix a, Matrix b){
    Matrix result (a.n, a.m);

    int i,j;

    for(i = 0; i < a.n; i++){
        for(j = 0; j < a.m; j++){
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    return result;
}

Matrix multiply(Matrix a, Matrix b) {
    Matrix result(a.n, b.m);
    
    for (int i = 0; i < a.n; i++) {
        for (int j = 0; j < b.m; j++) {
            result.a[i][j] = 0;
            for (int k = 0; k < a.m; k++) {
                result.a[i][j] += a.a[i][k] * b.a[k][j];
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

    delta = hadamard (substract(input, output), sigmoid_dx_mat(l[n-1]));

    nabla_b[n - 2].add(delta);
    nabla_w[n - 2].add(multiply(transpose(l[n - 2]), delta));

    for(i = n - 3; i >= 0; i--){
        delta = multiply(delta, transpose(w[i+1]));

        delta = hadamard(delta, sigmoid_dx_mat(l[i+1]));

        nabla_b[i].add(delta);
        nabla_w[i].add(multiply(transpose(l[i]), delta));
    }
}

void Neural_Network::train(std::vector<Matrix> inputs, std::vector<Matrix> outputs) {
    int i, j, k;

    for (i = 0; i < n - 1; i++) {
        nabla_w[i].zero();
        nabla_b[i].zero();
    }

    for (i = 0; i < static_cast<int>(inputs.size()); i++) {
        backpropagation(inputs[i], outputs[i]);
    }

    for (i = 0; i < n - 1; i++) {
        for (j = 0; j < nabla_w[i].n; j++) {
            for (k = 0; k < nabla_w[i].m; k++) {
                nabla_w[i][j][k] /= static_cast<double>(inputs.size());
                w[i][j][k] -= learning_rate * nabla_w[i][j][k];
            }
        }

        for (j = 0; j < nabla_b[i].n; j++) {
            for (k = 0; k < nabla_b[i].m; k++) {
                nabla_b[i][j][k] /= static_cast<double>(inputs.size());
                b[i][j][k] -= learning_rate * nabla_b[i][j][k];
            }
        }
    }
}

void time_stamp (){
    std::cerr << "Time: "<< (int)(clock() * 1000.0/CLOCKS_PER_SEC) << "ms. "<< std::endl;
}

void parse_training_data (){
    std::ifstream data_in ("train.csv");

    std::string trash;

    std::vector<int> v;

    // This is for reading the data from the MNIST database
    // That's why it is 1 x 784 (28x28)
    Matrix input(1,784);
    // Here, the 10 dimensions, columns contain the probabilities
    Matrix output (1, 10); 

    train_input.reserve(42000);
    train_output.reserve(42000);

    data_in >> trash;

    int i,j;

    for(i = 0; i < 42000; i++){
        data_in >> trash;

        v = split(trash);

        output.zero();
        output[0][v[0]] = 1.0f;

        for(j = 1; j < 785; j++){
            input[0][j-1] = v[j] / 255.0;
        }
        train_input.push_back(input);
        train_output.push_back(output);
    }
    std::cout << "Training data loaded "<< std::endl;
    time_stamp();
}

std::vector<int> split (std::string x){
    int i, curr = 0;
    std::vector<int> res;

    for(i = 0; i < (int)x.size(); i++){
        if(x[i] == ','){
            res.push_back(curr);
            curr = 0;
        }else{
            curr *= 10;
            curr += x[i] - '0';
        }
    }

    res.push_back(curr);

    return res;
}