#include "image_handler.h"
#include "Utils.h"

#include <iostream>

int main() {
    std::string image_path1 = "image1.bmp";
    std::string image_path2 = "image2.bmp";

    Image test_image1(image_path1);
    Image test_image2(image_path2);

    test_image1.fulfill_matrix();
    test_image2.fulfill_matrix();

    double threshold = 0.90;
    get_prediction(test_image1, test_image2, threshold);

    return 0;
}
