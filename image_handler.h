#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include <string>
#include <fstream>
#include <vector>

class Image {
public:
    Image(std::string path);
    Image(const Image& other) = delete; // Copy constructor deleted
    Image& operator=(const Image& other) = delete; // Copy assignment operator deleted
    Image(Image&& other) noexcept; // Move constructor
    Image& operator=(Image&& other) noexcept; // Move assignment operator
    ~Image();

    void fulfill_matrix();
    std::vector<std::vector<char>> get_matrix ();

private:
    std::vector<std::vector<char>> image_matrix;
    std::ifstream image;
    std::string path;
};

#endif // IMAGE_HANDLER_H