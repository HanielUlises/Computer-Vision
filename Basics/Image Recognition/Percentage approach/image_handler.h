#ifndef IMAGE_HANDLER_H
#define IMAGE_HANDLER_H

#include <vector>
#include <string>
#include <fstream>

class Image {
public:
    Image(std::string path);
    ~Image();
    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    std::vector<std::vector<char>> get_matrix() const;

    void fulfill_matrix();

private:
    std::string path;
    std::ifstream image;
    std::vector<std::vector<char>> image_matrix;
};

int calculate_bmp_width(const std::string& filename);

#endif // IMAGE_HANDLER_H
