#include <stdio.h>
#include <stdlib.h>
#include "perceptron.h"
#include "mnist.h"

#define MODEL_FILE "perceptron_model.dat"
#define TEST_IMAGES_FILE "t10k-images-idx3-ubyte"
#define TEST_LABELS_FILE "t10k-labels-idx1-ubyte"

int main() {
    Perceptron *p = create_perceptron(28 * 28, 10);
    load_perceptron(p, MODEL_FILE);

    MNIST_Data *test_data = load_mnist_images(TEST_IMAGES_FILE);
    uint8_t *test_labels = load_mnist_labels(TEST_LABELS_FILE);

    if (!test_data || !test_labels) {
        fprintf(stderr, "Failed to load MNIST test data\n");
        return 1;
    }

    int correct = 0;
    for (int i = 0; i < test_data->num_images; i++) {
        double input[28 * 28];
        for (int j = 0; j < 28 * 28; j++) {
            input[j] = test_data->images[i][j] / 255.0;
        }
        int predicted = predict_perceptron(p, input);
        if (predicted == test_labels[i]) {
            correct++;
        }
    }

    printf("Accuracy: %.2f%%\n", (correct / (double)test_data->num_images) * 100.0);

    free_perceptron(p);
    free_mnist_data(test_data);
    free(test_labels);

    return 0;
}
