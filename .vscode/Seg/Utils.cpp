#include "Utils.h"

using namespace cv;

void get_histogram() {
    std::string im_path = "Resources/miau.jpg";

    Mat img;
    std::vector<Mat> bgr_planes;

    img = imread(im_path, IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return;
    }

    split(img, bgr_planes);

    int hist_size = 256;
    float range[] = { 0, 256 };
    const float* hist_range[] = { range };

    bool uniform = true;
    bool accumulate = false;

    Mat b_hist, g_hist, r_hist;

    // Calculate histograms for the B, G, and R channels
    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &hist_size, hist_range, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &hist_size, hist_range, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &hist_size, hist_range, uniform, accumulate);

    // Normalize the histograms to use the full height of the image
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / hist_size);

    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    for (int i = 1; i < hist_size; i++) {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
            Point(bin_w * i, hist_h - cvRound(b_hist.at<float>(i))),
            Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
            Point(bin_w * i, hist_h - cvRound(g_hist.at<float>(i))),
            Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
            Point(bin_w * i, hist_h - cvRound(r_hist.at<float>(i))),
            Scalar(0, 0, 255), 2, 8, 0);
    }

    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", img);
    namedWindow("Histogram", WINDOW_AUTOSIZE);
    imshow("Histogram", histImage);

    waitKey(0);
}

void get_LUT() {
    std::string im_path = "Resources/miau.jpg";
    int selection;

    Mat img; Mat lut_img(1, 256, CV_8U);

    img = imread(im_path, IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return;
    }
    std::cout << "Apply function" << std::endl;
    std::cin >> selection;
    switch (selection) {
    case 1:
        // Apply inverse function
        for (int i = 0; i < 256; i++) {
            lut_img.at<uchar>(i) = 255 - i;
        }
        break;
    case 2:
        // Apply  cuadratic function
        for (int i = 0; i < 256; i++) {
            lut_img.at<uchar>(i) = pow((float)(i) * 255, (float)(1/2.0));
        }
        break;
    case 3:
        // Apply cubic function
        for (int i = 0; i < 256; i++) {
            lut_img.at<uchar>(i) = pow((float)(i), (float)(3.0)) / (255 * 255);
        }
        break;
    default:
        break;
    }
    

    LUT(img, lut_img, img);
    namedWindow("Original", WINDOW_AUTOSIZE);
    imshow("Original", img);

    waitKey(0);
}