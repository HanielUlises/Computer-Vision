#ifndef MNIST_H
#define MNIST_H

#include <stdint.h>

typedef struct {
    int num_images;
    int rows;
    int cols;
    uint8_t **images;
    uint8_t *labels;
} MNIST_Data;

MNIST_Data* load_mnist_images(const char *file_name);
uint8_t* load_mnist_labels(const char *file_name);
void free_mnist_data(MNIST_Data *data);

#endif // MNIST_H
