#include <iostream>
#include "dataPath.hpp"
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

int maxScaleUp = 100;
int scaleFactor = 1;
int scaleType = 0;
int maxType = 1;
Mat im;

string windowName = "Resize Image";
string trackbarValue = "Scale";
string trackbarType = "Type: \n 0: Scale Up \n 1: Scale Down";

void scaleImage(int, void*);

int main() {
    // Load an image
    im = imread("truth.png");

    // Create a window to display results
    namedWindow(windowName, WINDOW_AUTOSIZE);

    createTrackbar(trackbarValue, windowName, &scaleFactor, maxScaleUp, scaleImage);
    createTrackbar(trackbarType, windowName, &scaleType, maxType, scaleImage);

    scaleImage(25, 0);

    while (true) {
        int c;
        c = waitKey(20);
        if (static_cast<char>(c) == 27)
            break;
    }

    return 0;
}

// Callback function
void scaleImage(int, void*) {
    double scaleFactorDouble;
    if (scaleType == 0) {
        scaleFactorDouble = 1 + scaleFactor / 100.0;
    } else {
        scaleFactorDouble = 1 - 0.5 * (scaleFactor / 100.0);
    }
    Mat scaledImage;
    resize(im, scaledImage, Size(), scaleFactorDouble, scaleFactorDouble, INTER_LINEAR);
    imshow(windowName, scaledImage);
}