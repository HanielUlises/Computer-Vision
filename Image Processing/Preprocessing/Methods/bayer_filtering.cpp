#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include <stdexcept>

std::vector<uint8_t> read_raw_image(const std::string& filename, int width, int height) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    std::vector<uint8_t> data(width * height);
    file.read(reinterpret_cast<char*>(data.data()), data.size());
    if (file.gcount() != data.size()) {
        throw std::runtime_error("File size mismatch");
    }
    return data;
}

void write_ppm(const std::string& filename, const std::vector<uint8_t>& rgb_data, int width, int height) {
    std::ofstream file(filename);
    if (!file) {
        throw std::runtime_error("Failed to create file: " + filename);
    }
    file << "P6\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(rgb_data.data()), rgb_data.size());
}

// (Adaptive Homogeneity-Directed) demosaicing.
// It interpolates greens first, then reds/blues based on homogeneity in directions.

void demosaic_ahd(const std::vector<uint8_t>& bayer, int width, int height, std::vector<uint8_t>& rgb) {
    rgb.resize(width * height * 3);

    // Row even: R G R G ...
    // Row odd:  G B G B ...

    int pad_width = width + 2;
    int pad_height = height + 2;
    std::vector<uint8_t> padded_bayer(pad_width * pad_height, 0);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            padded_bayer[(y + 1) * pad_width + (x + 1)] = bayer[y * width + x];
        }
    }

    for (int x = 0; x < pad_width; ++x) {
        padded_bayer[0 * pad_width + x] = padded_bayer[1 * pad_width + x];
        padded_bayer[(pad_height - 1) * pad_width + x] = padded_bayer[(pad_height - 2) * pad_width + x];
    }

    for (int y = 0; y < pad_height; ++y) {
        padded_bayer[y * pad_width + 0] = padded_bayer[y * pad_width + 1];
        padded_bayer[y * pad_width + (pad_width - 1)] = padded_bayer[y * pad_width + (pad_width - 2)];
    }


    std::vector<uint8_t> green(width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if ((x % 2 == 0 && y % 2 == 0) || (x % 2 == 1 && y % 2 == 1)) {

                int north = padded_bayer[y * pad_width + (x + 1)];
                int south = padded_bayer[(y + 2) * pad_width + (x + 1)];
                int west = padded_bayer[(y + 1) * pad_width + x];
                int east = padded_bayer[(y + 1) * pad_width + (x + 2)];

                // Gradients for directions
                int grad_h = std::abs(west - east) + std::abs(2 * padded_bayer[(y + 1) * pad_width + (x + 1)] - west - east);
                int grad_v = std::abs(north - south) + std::abs(2 * padded_bayer[(y + 1) * pad_width + (x + 1)] - north - south);

                if (grad_h < grad_v) {
                    green[y * width + x] = (west + east) / 2;
                } else if (grad_v < grad_h) {
                    green[y * width + x] = (north + south) / 2;
                } else {
                    green[y * width + x] = (north + south + west + east) / 4;
                }
            } else {
                // Green positions
                green[y * width + x] = padded_bayer[(y + 1) * pad_width + (x + 1)];
            }
        }
    }

    std::vector<uint8_t> padded_green(pad_width * pad_height, 0);

    // Fill inner part
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            padded_green[(y + 1) * pad_width + (x + 1)] = green[y * width + x];
        }
    }

    // Mirror top and bottom and mirror left and right
    for (int x = 0; x < pad_width; ++x) {
        padded_green[0 * pad_width + x] = padded_green[1 * pad_width + x];
        padded_green[(pad_height - 1) * pad_width + x] = padded_green[(pad_height - 2) * pad_width + x];
    }

    for (int y = 0; y < pad_height; ++y) {
        padded_green[y * pad_width + 0] = padded_green[y * pad_width + 1];
        padded_green[y * pad_width + (pad_width - 1)] = padded_green[y * pad_width + (pad_width - 2)];
    }

    // Chroma interpolation
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int rgb_idx = idx * 3;
            int pad_y = y + 1;
            int pad_x = x + 1;

            if (y % 2 == 0 && x % 2 == 0) { // Red position
                rgb[rgb_idx + 0] = padded_bayer[pad_y * pad_width + pad_x]; // R
                rgb[rgb_idx + 1] = padded_green[pad_y * pad_width + pad_x]; // G

                // Interpolate blue at red position (average neighbors with green correction)
                int blue_val = 0;
                blue_val += padded_bayer[(pad_y - 1) * pad_width + (pad_x - 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y - 1) * pad_width + (pad_x - 1)]);
                blue_val += padded_bayer[(pad_y - 1) * pad_width + (pad_x + 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y - 1) * pad_width + (pad_x + 1)]);
                blue_val += padded_bayer[(pad_y + 1) * pad_width + (pad_x - 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y + 1) * pad_width + (pad_x - 1)]);
                blue_val += padded_bayer[(pad_y + 1) * pad_width + (pad_x + 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y + 1) * pad_width + (pad_x + 1)]);
                rgb[rgb_idx + 2] = std::clamp(blue_val / 4, 0, 255);
            } else if (y % 2 == 1 && x % 2 == 1) { // Blue position
                rgb[rgb_idx + 2] = padded_bayer[pad_y * pad_width + pad_x]; // B
                rgb[rgb_idx + 1] = padded_green[pad_y * pad_width + pad_x]; // G

                // Interpolate red at blue position
                int red_val = 0;
                red_val += padded_bayer[(pad_y - 1) * pad_width + (pad_x - 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y - 1) * pad_width + (pad_x - 1)]);
                red_val += padded_bayer[(pad_y - 1) * pad_width + (pad_x + 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y - 1) * pad_width + (pad_x + 1)]);
                red_val += padded_bayer[(pad_y + 1) * pad_width + (pad_x - 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y + 1) * pad_width + (pad_x - 1)]);
                red_val += padded_bayer[(pad_y + 1) * pad_width + (pad_x + 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y + 1) * pad_width + (pad_x + 1)]);
                rgb[rgb_idx + 0] = std::clamp(red_val / 4, 0, 255);
            } else if (y % 2 == 0 && x % 2 == 1) { // Green on even row (between R)
                rgb[rgb_idx + 1] = padded_bayer[pad_y * pad_width + pad_x]; // G

                // Interpolate red horizontally
                int red_val = 0;
                red_val += padded_bayer[pad_y * pad_width + (pad_x - 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[pad_y * pad_width + (pad_x - 1)]);
                red_val += padded_bayer[pad_y * pad_width + (pad_x + 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[pad_y * pad_width + (pad_x + 1)]);
                rgb[rgb_idx + 0] = std::clamp(red_val / 2, 0, 255);

                // Interpolate blue vertically
                int blue_val = 0;
                blue_val += padded_bayer[(pad_y - 1) * pad_width + pad_x] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y - 1) * pad_width + pad_x]);
                blue_val += padded_bayer[(pad_y + 1) * pad_width + pad_x] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y + 1) * pad_width + pad_x]);
                rgb[rgb_idx + 2] = std::clamp(blue_val / 2, 0, 255);
            } else { // Green on odd row (between B)
                rgb[rgb_idx + 1] = padded_bayer[pad_y * pad_width + pad_x]; // G

                // Interpolate blue horizontally
                int blue_val = 0;
                blue_val += padded_bayer[pad_y * pad_width + (pad_x - 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[pad_y * pad_width + (pad_x - 1)]);
                blue_val += padded_bayer[pad_y * pad_width + (pad_x + 1)] + (padded_green[pad_y * pad_width + pad_x] - padded_green[pad_y * pad_width + (pad_x + 1)]);
                rgb[rgb_idx + 2] = std::clamp(blue_val / 2, 0, 255);

                // Interpolate red vertically
                int red_val = 0;
                red_val += padded_bayer[(pad_y - 1) * pad_width + pad_x] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y - 1) * pad_width + pad_x]);
                red_val += padded_bayer[(pad_y + 1) * pad_width + pad_x] + (padded_green[pad_y * pad_width + pad_x] - padded_green[(pad_y + 1) * pad_width + pad_x]);
                rgb[rgb_idx + 0] = std::clamp(red_val / 2, 0, 255);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <input_raw> <output_ppm> <width> <height>\n";
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    int width = std::stoi(argv[3]);
    int height = std::stoi(argv[4]);

    try {
        auto bayer = read_raw_image(input_file, width, height);
        std::vector<uint8_t> rgb;
        demosaic_ahd(bayer, width, height, rgb);
        write_ppm(output_file, rgb, width, height);
        std::cout << "Demosaiced image saved to " << output_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}