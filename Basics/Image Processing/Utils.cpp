#include "Utils.h"

using namespace cv;

void get_histogram() {
    std::string im_path = "Resources/fruits.bmp";

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
    std::string im_path = "Resources/fruits.bmp";
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
    namedWindow("Modified", WINDOW_AUTOSIZE);
    imshow("Original", img);

    waitKey(0);
}

void change_color_space(){
    std::string im_path = "Resources/fruits.bmp";
    Mat imgRGB, imgHSV;

    imgRGB = imread(im_path, IMREAD_COLOR);
    if (imgRGB.empty()) {
        std::cout << "Could not open or find the image" << std::endl;
        return;
    }
    // Split image into three sub-images (A, V and R)
    std::vector<Mat> bgr_planes;
    split(imgRGB, bgr_planes);

    // Change space color 
    cvtColor(imgRGB, imgHSV, COLOR_BGR2HSV);
    std::vector<Mat> hsv_planes;
    split(imgHSV, hsv_planes);

    

    std::string input, input_2;
    std::cout << "RGB? or HSV? " << std::endl;
    std::cin >> input;

    if (input == "RGB") {
        std::cout << "Full image or channels?" << std::endl;
        std::cin >> input_2;
        if (input_2 == "Full") {
            namedWindow("RGB_Image", WINDOW_AUTOSIZE);
            imshow("RGB_Image", imgRGB);
        }
        else if (input_2 == "Channels") {
            namedWindow("B", WINDOW_AUTOSIZE);
            imshow("B", bgr_planes[0]);
            namedWindow("G", WINDOW_AUTOSIZE);
            imshow("G", bgr_planes[1]);
            namedWindow("R", WINDOW_AUTOSIZE);
            imshow("R", bgr_planes[2]);
        }
    }
    else if (input == "HSV") {
        std::cout << "Full image or channels?" << std::endl;
        std::cin >> input_2;
        if (input == "Full") {
            namedWindow("HSV_Image", WINDOW_AUTOSIZE);
            imshow("HSV_Image", imgHSV);
        }
        else if (input == "Channels") {
            namedWindow("H", WINDOW_AUTOSIZE);
            imshow("H", hsv_planes[0]);
            namedWindow("S", WINDOW_AUTOSIZE);
            imshow("S", hsv_planes[1]);
            namedWindow("V", WINDOW_AUTOSIZE);
            imshow("V", hsv_planes[2]);
        }
    }
    waitKey(0);
}

void addition() {
    std::string im_path1 = "Resources/image1.bmp";
    std::string im_path2 = "Resources/image2.bmp";

    Mat img1 = imread(im_path1, IMREAD_COLOR);
    Mat img2 = imread(im_path2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cout << "Could not open or find one of the images" << std::endl;
        return;
    }

    // Ensure both images are of the same size and type
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cout << "Images have different sizes or types" << std::endl;
        return;
    }

    Mat result = Mat::zeros(img1.size(), img1.type());

    // Addition operation by iterating over each pixel
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            for (int c = 0; c < img1.channels(); c++) {
                result.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(img1.at<Vec3b>(y, x)[c] + img2.at<Vec3b>(y, x)[c]);
            }
        }
    }

    namedWindow("Addition Result", WINDOW_AUTOSIZE);
    imshow("Addition Result", result);
    waitKey(0);
}

void subtraction() {
    std::string im_path1 = "Resources/image1.bmp";
    std::string im_path2 = "Resources/image2.bmp";

    Mat img1 = imread(im_path1, IMREAD_COLOR);
    Mat img2 = imread(im_path2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cout << "Could not open or find one of the images" << std::endl;
        return;
    }

    // Ensure both images are of the same size and type
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cout << "Images have different sizes or types" << std::endl;
        return;
    }

    Mat result = Mat::zeros(img1.size(), img1.type());

    // Subtraction operation by iterating over each pixel
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            for (int c = 0; c < img1.channels(); c++) {
                result.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(img1.at<Vec3b>(y, x)[c] - img2.at<Vec3b>(y, x)[c]);
            }
        }
    }

    namedWindow("Subtraction Result", WINDOW_AUTOSIZE);
    imshow("Subtraction Result", result);
    waitKey(0);
}

void multiplication() {
    std::string im_path1 = "Resources/image1.bmp";
    std::string im_path2 = "Resources/image2.bmp";

    Mat img1 = imread(im_path1, IMREAD_COLOR);
    Mat img2 = imread(im_path2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cout << "Could not open or find one of the images" << std::endl;
        return;
    }

    // Ensure both images are of the same size and type
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cout << "Images have different sizes or types" << std::endl;
        return;
    }

    Mat result = Mat::zeros(img1.size(), img1.type());

    // Multiplication operation by iterating over each pixel
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            for (int c = 0; c < img1.channels(); c++) {
                result.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((img1.at<Vec3b>(y, x)[c] * img2.at<Vec3b>(y, x)[c]) / 255);
            }
        }
    }

    namedWindow("Multiplication Result", WINDOW_AUTOSIZE);
    imshow("Multiplication Result", result);
    waitKey(0);
}

void division() {
    std::string im_path1 = "Resources/image1.bmp";
    std::string im_path2 = "Resources/image2.bmp";

    Mat img1 = imread(im_path1, IMREAD_COLOR);
    Mat img2 = imread(im_path2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cout << "Could not open or find one of the images" << std::endl;
        return;
    }

    // Ensure both images are of the same size and type
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cout << "Images have different sizes or types" << std::endl;
        return;
    }

    Mat result = Mat::zeros(img1.size(), img1.type());

    // Division operation by iterating over each pixel
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            for (int c = 0; c < img1.channels(); c++) {
                uchar denom = img2.at<Vec3b>(y, x)[c];
                if (denom != 0) {  // Avoid division by zero
                    result.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((img1.at<Vec3b>(y, x)[c] * 255) / denom);
                }
                else {
                    result.at<Vec3b>(y, x)[c] = 0;
                }
            }
        }
    }

    namedWindow("Division Result", WINDOW_AUTOSIZE);
    imshow("Division Result", result);
    waitKey(0);
}

void and_img() {
    std::string im_path1 = "Resources/image1.bmp";
    std::string im_path2 = "Resources/image2.bmp";

    Mat img1 = imread(im_path1, IMREAD_COLOR);
    Mat img2 = imread(im_path2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cout << "Could not open or find one of the images" << std::endl;
        return;
    }

    // Ensure both images are of the same size and type
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cout << "Images have different sizes or types" << std::endl;
        return;
    }

    Mat result = Mat::zeros(img1.size(), img1.type());

    // AND operation by iterating over each pixel
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            for (int c = 0; c < img1.channels(); c++) {
                result.at<Vec3b>(y, x)[c] = img1.at<Vec3b>(y, x)[c] & img2.at<Vec3b>(y, x)[c];
            }
        }
    }

    namedWindow("AND Result", WINDOW_AUTOSIZE);
    imshow("AND Result", result);
    waitKey(0);
}

void or_img() {
    std::string im_path1 = "Resources/image1.bmp";
    std::string im_path2 = "Resources/image2.bmp";

    Mat img1 = imread(im_path1, IMREAD_COLOR);
    Mat img2 = imread(im_path2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cout << "Could not open or find one of the images" << std::endl;
        return;
    }

    // Ensure both images are of the same size and type
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cout << "Images have different sizes or types" << std::endl;
        return;
    }

    Mat result = Mat::zeros(img1.size(), img1.type());

    // OR operation by iterating over each pixel
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            for (int c = 0; c < img1.channels(); c++) {
                result.at<Vec3b>(y, x)[c] = img1.at<Vec3b>(y, x)[c] | img2.at<Vec3b>(y, x)[c];
            }
        }
    }

    namedWindow("OR Result", WINDOW_AUTOSIZE);
    imshow("OR Result", result);
    waitKey(0);
}

void xor_img() {
    std::string im_path1 = "Resources/image1.bmp";
    std::string im_path2 = "Resources/image2.bmp";

    Mat img1 = imread(im_path1, IMREAD_COLOR);
    Mat img2 = imread(im_path2, IMREAD_COLOR);

    if (img1.empty() || img2.empty()) {
        std::cout << "Could not open or find one of the images" << std::endl;
        return;
    }

    // Ensure both images are of the same size and type
    if (img1.size() != img2.size() || img1.type() != img2.type()) {
        std::cout << "Images have different sizes or types" << std::endl;
        return;
    }

    Mat result = Mat::zeros(img1.size(), img1.type());

    // XOR operation by iterating over each pixel
    for (int y = 0; y < img1.rows; y++) {
        for (int x = 0; x < img1.cols; x++) {
            for (int c = 0; c < img1.channels(); c++) {
                result.at<Vec3b>(y, x)[c] = img1.at<Vec3b>(y, x)[c] ^ img2.at<Vec3b>(y, x)[c];
            }
        }
    }

    namedWindow("XOR Result", WINDOW_AUTOSIZE);
    imshow("XOR Result", result);
    waitKey(0);
}

void correlation() {
    std::string original_image = "Resources/image_1.bmp";
    // Objective to be located
    std::string template_image = "Resoures/template_test";

    Mat src = imread(original_image);
    Mat templ = imread(template_image);

    if (!src.data || !templ.data) {
        std::cout << "Couldn't load the images " << std::endl;
        exit(1);
    }

    Mat dst;

    // Memory allocation for each mode
    int i_width = src.cols - templ.cols + 1;
    int i_height = src.rows - templ.rows + 1;

    dst.create(i_width, i_height, CV_32FC1);

    // Method by coefficient correlation
    int match_method = TM_CCOEFF_NORMED;

    // Correlation
    matchTemplate(src, templ, dst, match_method);
    normalize(dst, dst, 0, 1, NORM_MINMAX, -1, Mat());

    double min_val, max_val;
    
    Point min_loc, max_loc;
    Point match_loc;

    minMaxLoc(dst, &min_val, &max_val, &min_loc, &max_loc, Mat());

    std::cout << "Min: " << min_val << "Max: " << max_val << std::endl;

    if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED) {
        match_loc = min_loc;
    }
    else {
        match_loc = max_loc;
    }

    rectangle(src, match_loc, Point(match_loc.x + templ.cols, match_loc.y + templ.rows), Scalar(255, 0, 0), 4, 8, 0);
    rectangle(dst, match_loc, Point(match_loc.x + templ.cols, match_loc.y + templ.rows), Scalar::all(0), 4, 8, 0);

    imshow("Source", src);
    imshow("Result", dst);
    imshow("Template", templ);

    waitKey(0);

}