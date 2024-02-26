#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include <string>
#include <fstream>
#include <vector>

class Image {
public:
    Image(std::string path);
    ~Image();

    Image(const Image&) = delete; // Deleted copy constructor
    Image& operator=(const Image&) = delete; // Deleted copy assignment operator

    Image(Image&& other) noexcept; // Move constructor
    Image& operator=(Image&& other) noexcept; // Move assignment operator

    void fulfill_matrix();
    std::vector<std::vector<char>> get_matrix () const;

private:
    std::vector<std::vector<char>> image_matrix;
    std::ifstream image;
    std::string path;
};

#endif // IMAGE_HANDLER_H
