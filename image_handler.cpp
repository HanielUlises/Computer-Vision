#include "image_handler.h"
#include <iostream>
#include <utility>
#include <memory>

Image::Image(std::string path): path(std::move(path)) {
    image.open(this->path, std::ifstream::in);
}

Image::Image(Image&& other) noexcept : image_matrix(std::move(other.image_matrix)), path(std::move(other.path)) {
    other.image.close(); // Ensure we leave the moved-from object in a valid state
    image.open(path, std::ifstream::in); // Open the file for the newly moved object
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        image.close(); // Close current file if open

        image_matrix = std::move(other.image_matrix);
        path = std::move(other.path);
        image.open(path, std::ifstream::in); // Open the file for the newly moved object

        // Clear the moved-from object to a valid state
        other.image_matrix.clear();
    }
    return *this;
}

Image::~Image() {
    if (image.is_open()) {
        image.close();
    }
}

void Image::fulfill_matrix() {
    // Ensure file is open
    if (!image.is_open()) {
        std::cerr << "Error: Unable to open file " << path << std::endl;
        return;
    }

    image_matrix.clear(); // Clear any existing data
    char pixel;
    std::vector<char> row;
    while (image.get(pixel)) {
        if (pixel == '\n') {
            image_matrix.push_back(row);
            row.clear();
        } else {
            row.push_back(pixel);
        }
    }

    // No need to close the file here if you plan to use it later
}

std::vector<std::vector<char>> Image::get_matrix() {
    return image_matrix;
}
