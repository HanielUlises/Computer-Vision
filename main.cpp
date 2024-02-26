#include "image_handler.h"
#include "Utils.h"

#include <iostream>

int main() {
    std::string image_path1 = "./imagen.bmp";
    std::string image_path2 = "./imagen2.bmp";

    Image test_image1(image_path1);
    Image test_image2(image_path2);

    double threshold = 150;
    get_prediction(test_image1, test_image2, threshold);

    return 0;
}
