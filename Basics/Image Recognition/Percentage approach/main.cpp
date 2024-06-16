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

    double threshold_percentage = 0.80; // 80% similarity
    double threshold_weight = 1000; // Example threshold for exhaustive search

    std::cout << "Prediction based on percentage similarity:" << std::endl;
    get_prediction_percentage(test_image1, test_image2, threshold_percentage);

    std::cout << "Prediction based on exhaustive search:" << std::endl;
    get_prediction_exhaustive(test_image1, test_image2, threshold_weight);

    return 0;
}
