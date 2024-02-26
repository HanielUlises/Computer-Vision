#include "image_handler.h"
#include <iostream>
#include <utility>
#include <fstream>

int calculate_bmp_width(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return -1;
    }

    file.seekg(18); 

    uint32_t width;
    file.read(reinterpret_cast<char*>(&width), sizeof(width));

    file.close();

    return static_cast<int>(width);
}

// Constructor
Image::Image(std::string path): path(std::move(path)) {
    image.open(this->path, std::ifstream::in);
    if (image.is_open()) {
        fulfill_matrix();
    }
}

// Destructor
Image::~Image() {
    if (image.is_open()) {
        image.close();
    }
}

// Move constructor
Image::Image(Image&& other) noexcept : image_matrix(std::move(other.image_matrix)), path(std::move(other.path)) {
    other.image.close(); // Ensure we leave the moved-from object in a valid state
    image.open(path, std::ifstream::in); // Open the file for the newly moved object
}

// Move assignment operator
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

std::vector<std::vector<char>> Image::get_matrix() {
    return image_matrix;
}

void Image::fulfill_matrix() {
    if (!image.is_open()) {
        std::cerr << "Error: Unable to open file " << path << std::endl;
        return;
    }

    image_matrix.clear(); // Clear any existing data

    // Determine the width of the image (number of columns)
    // By knowing the width of the image, one can read that many characters per row.
    int image_width = calculate_bmp_width(path);
    char pixel;

    std::vector<char> row;
    int pixel_count = 0;
    while (image.get(pixel)) {
        row.push_back(pixel);
        pixel_count++;

        // Check if we have read a complete row
        if (pixel_count % image_width == 0) {
            image_matrix.push_back(row);
            row.clear();
        }
    }
}