#include "mnist.h"
#include <stdio.h>
#include <stdlib.h>

MNIST_Data* load_mnist_images(const char *file_name) {
    FILE *file = fopen(file_name, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Read the header information
    int magic_number = 0;
    int num_images = 0;
    int rows = 0;
    int cols = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    fread(&num_images, sizeof(num_images), 1, file);
    fread(&rows, sizeof(rows), 1, file);
    fread(&cols, sizeof(cols), 1, file);

    // Conversion from big-endian to host byte order
    magic_number = __builtin_bswap32(magic_number);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (magic_number != 2051) {
        fprintf(stderr, "Invalid MNIST image file!\n");
        fclose(file);
        return NULL;
    }

    MNIST_Data *data = (MNIST_Data*)malloc(sizeof(MNIST_Data));
    data->num_images = num_images;
    data->rows = rows;
    data->cols = cols;
    data->images = (uint8_t**)malloc(num_images * sizeof(uint8_t*));

    for (int i = 0; i < num_images; i++) {
        data->images[i] = (uint8_t*)malloc(rows * cols * sizeof(uint8_t));
        fread(data->images[i], sizeof(uint8_t), rows * cols, file);
    }

    fclose(file);
    return data;
}

uint8_t* load_mnist_labels(const char *file_name) {
    FILE *file = fopen(file_name, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Read the header information
    int magic_number = 0;
    int num_labels = 0;

    fread(&magic_number, sizeof(magic_number), 1, file);
    fread(&num_labels, sizeof(num_labels), 1, file);

    // Conversion from big-endian to host byte order
    magic_number = __builtin_bswap32(magic_number);
    num_labels = __builtin_bswap32(num_labels);

    if (magic_number != 2049) {
        fprintf(stderr, "Invalid MNIST label file!\n");
        fclose(file);
        return NULL;
    }

    uint8_t *labels = (uint8_t*)malloc(num_labels * sizeof(uint8_t));
    fread(labels, sizeof(uint8_t), num_labels, file);

    fclose(file);
    return labels;
}

void free_mnist_data(MNIST_Data *data) {
    for (int i = 0; i < data->num_images; i++) {
        free(data->images[i]);
    }
    free(data->images);
    free(data->labels);
    free(data);
}
