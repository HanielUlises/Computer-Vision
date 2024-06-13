#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <SDL2/SDL.h>
#include "perceptron.h"
#include "mnist.h"

#define TRAIN_IMAGES_FILE "train-images-idx3-ubyte"
#define TRAIN_LABELS_FILE "train-labels-idx1-ubyte"
#define TEST_IMAGES_FILE  "t10k-images-idx3-ubyte"
#define TEST_LABELS_FILE  "t10k-labels-idx1-ubyte"

#define EPOCHS 1000
#define LEARNING_RATE 0.1
#define MODEL_FILE "perceptron_model.dat"
#define WINDOW_WIDTH 280
#define WINDOW_HEIGHT 280
#define GRID_SIZE 10

void process_image(SDL_Renderer *renderer, double *input, int width, int height) {
    SDL_Surface *surface = SDL_CreateRGBSurfaceWithFormat(0, width, height, 32, SDL_PIXELFORMAT_ARGB8888);
    SDL_RenderReadPixels(renderer, NULL, SDL_PIXELFORMAT_ARGB8888, surface->pixels, surface->pitch);

    uint8_t *pixels = (uint8_t *)surface->pixels;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int offset = (y * width + x) * 4;
            uint8_t r = pixels[offset + 2];
            uint8_t g = pixels[offset + 1];
            uint8_t b = pixels[offset];
            double gray = 0.299 * r + 0.587 * g + 0.114 * b;
            input[y * width + x] = gray / 255.0;
        }
    }

    SDL_FreeSurface(surface);
}

void draw_and_predict(Perceptron* p) {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "Could not initialize SDL: %s\n", SDL_GetError());
        return;
    }

    SDL_Window *window = SDL_CreateWindow("Draw a Digit", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    if (!window) {
        fprintf(stderr, "Could not create window: %s\n", SDL_GetError());
        SDL_Quit();
        return;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "Could not create renderer: %s\n", SDL_GetError());
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);

    int running = 1;
    int drawing = 0;
    SDL_Event event;

    while (running) {
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
                case SDL_QUIT:
                    running = 0;
                    break;
                case SDL_MOUSEBUTTONDOWN:
                    drawing = 1;
                    break;
                case SDL_MOUSEBUTTONUP:
                    drawing = 0;
                    break;
                case SDL_MOUSEMOTION:
                    if (drawing) {
                        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
                        SDL_Rect rect = { event.motion.x / GRID_SIZE * GRID_SIZE, event.motion.y / GRID_SIZE * GRID_SIZE, GRID_SIZE, GRID_SIZE };
                        SDL_RenderFillRect(renderer, &rect);
                        SDL_RenderPresent(renderer);
                    }
                    break;
                case SDL_KEYDOWN:
                    if (event.key.keysym.sym == SDLK_RETURN) {
                        // Process the image once the draw is completed 
                        // The input is then predicted based on the trained perceptron 
                        double input[28 * 28];
                        process_image(renderer, input, 28, 28);
                        int predicted = predict_perceptron(p, input);
                        printf("Predicted digit: %d\n", predicted);
                    }
                    break;
            }
        }
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s [train|draw]\n", argv[0]);
        return 1;
    }

    if (strcmp(argv[1], "train") == 0) {
        // Training mode
        MNIST_Data *train_data = load_mnist_images(TRAIN_IMAGES_FILE);
        uint8_t *train_labels = load_mnist_labels(TRAIN_LABELS_FILE);

        MNIST_Data *test_data = load_mnist_images(TEST_IMAGES_FILE);
        uint8_t *test_labels = load_mnist_labels(TEST_LABELS_FILE);

        if (!train_data || !train_labels || !test_data || !test_labels) {
            fprintf(stderr, "Failed to load MNIST data\n");
            return 1;
        }

        int input_size = train_data->rows * train_data->cols;
        int num_classes = 10;

        double **train_inputs = (double**)malloc(train_data->num_images * sizeof(double*));
        double **train_targets = (double**)malloc(train_data->num_images * sizeof(double*));
        for (int i = 0; i < train_data->num_images; i++) {
            train_inputs[i] = (double*)malloc(input_size * sizeof(double));
            train_targets[i] = (double*)calloc(num_classes, sizeof(double));

            for (int j = 0; j < input_size; j++) {
                train_inputs[i][j] = train_data->images[i][j] / 255.0;
            }
            train_targets[i][train_labels[i]] = 1.0;
        }

        Perceptron* p = create_perceptron(input_size, num_classes);
        train_perceptron(p, train_inputs, train_targets, train_data->num_images, EPOCHS, LEARNING_RATE);

        // Trained perceptron
        save_perceptron(p, MODEL_FILE);

        int correct = 0;
        for (int i = 0; i < test_data->num_images; i++) {
            double *input = (double*)malloc(input_size * sizeof(double));
            for (int j = 0; j < input_size; j++) {
                input[j] = test_data->images[i][j] / 255.0;
            }
            int predicted = predict_perceptron(p, input);
            if (predicted == test_labels[i]) {
                correct++;
            }
            free(input);
        }

        printf("Accuracy: %.2f%%\n", (correct / (double)test_data->num_images) * 100.0);

        free_perceptron(p);

        for (int i = 0; i < train_data->num_images; i++) {
            free(train_inputs[i]);
            free(train_targets[i]);
        }
        free(train_inputs);
        free(train_targets);

        free_mnist_data(train_data);
        free(train_labels);
        free_mnist_data(test_data);
        free(test_labels);
    } else if (strcmp(argv[1], "draw") == 0) {
        // Drawing mode
        Perceptron* p = create_perceptron(28 * 28, 10);
        load_perceptron(p, MODEL_FILE);
        draw_and_predict(p);
        free_perceptron(p);
    } else {
        fprintf(stderr, "Invalid mode. Use 'train' or 'draw'.\n");
        return 1;
    }

    return 0;
}
