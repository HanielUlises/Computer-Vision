#include <opencv2/opencv.hpp>
using namespace cv;

Mat pencil_sketch(Mat image, int arguments = 0) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Mat blurred;
    medianBlur(gray, blurred, 7);
    Mat edges;
    adaptiveThreshold(blurred, edges, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 2);
    Mat inverted_edges;
    bitwise_not(edges, inverted_edges);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat thickened_edges;
    dilate(inverted_edges, thickened_edges, kernel, Point(-1, -1), 2);
    Mat sketch;
    bitwise_not(thickened_edges, sketch);
    Mat pencil_sketch_image;
    cvtColor(sketch, pencil_sketch_image, COLOR_GRAY2BGR);
    return pencil_sketch_image;
}

Mat cartoonify(Mat image, int arguments = 0) {
    Mat cartoon_image;
    Mat smoothed;
    bilateralFilter(image, smoothed, 9, 75, 75);
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Mat blurred;
    medianBlur(gray, blurred, 7);
    Mat edges;
    adaptiveThreshold(blurred, edges, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 9, 2);
    Mat edges_colored;
    cvtColor(edges, edges_colored, COLOR_GRAY2BGR);
    cartoon_image = smoothed.clone();
    bitwise_and(smoothed, edges_colored, cartoon_image);
    return cartoon_image;
}

int main() {
    Mat image = imread("input.jpg");
    if (image.empty()) {
        return -1;
    }

    Mat pencil_sketch_result = pencil_sketch(image);
    Mat cartoon_result = cartoonify(image);

    imwrite("pencil_sketch_output.jpg", pencil_sketch_result);
    imwrite("cartoon_output.jpg", cartoon_result);

    return 0;
}