#include "image_handler.h"
#include <iostream>
#include <utility>
#include <fstream>

// Function to calculate BMP image width
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
    image.open(this->path, std::ifstream::binary);

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

// Method to get the image matrix
std::vector<std::vector<char>> Image::get_matrix() const {
    return image_matrix;
}

// Method to fulfill the image matrix
void Image::fulfill_matrix() {
    if (!image.is_open()) {
        std::cerr << "Error: Unable to open file " << path << std::endl;
        return;
    }

    image_matrix.clear();

    image.seekg(54, std::ios::beg); // Move to the pixel data start position

    int image_width = calculate_bmp_width(path);
    int padding = (4 - (image_width % 4)) % 4; // Calculate padding for each row
    char pixel;

    std::vector<char> row;
    for (int i = 0; i < image_width; ++i) {
        for (int j = 0; j < image_width; ++j) {
            if (image.get(pixel)) {
                row.push_back(pixel);
            }
        }
        image.seekg(padding, std::ios::cur); // Skip the padding
        image_matrix.push_back(row);
        row.clear();
    }
}
