#include <opencv2/opencv.hpp>
#include "embedding.hpp"

Embedding extract_embedding(const cv::Mat& img) {
    Embedding e(128);

    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, {16, 8});

    for (int i = 0; i < gray.total(); ++i)
        e.data[i] = gray.data[i] / 255.0f;

    e.normalize();
    return e;
}

int main() {
    cv::Mat img1 = cv::imread("img1.png");
    cv::Mat img2 = cv::imread("img2.png");

    auto e1 = extract_embedding(img1);
    auto e2 = extract_embedding(img2);

    std::cout << "Embedding distance: "
              << e1.norm() << std::endl;
}
