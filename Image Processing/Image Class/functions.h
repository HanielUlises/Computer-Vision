#ifndef FUNCTIONS_H
#define FUNCTION_H

#include "image.h"

#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>

#include<opencv2/opencv.hpp>

Image read(std::string path){
    cv::Mat image = cv::imread(path);
    if(image.empty()){
        std::cout << "Image file not found" << '\n';
    }
    Image img(image.rows, image.cols, image.channels());

    for(int i = 0; i < image.rows; i++){
        for(int j = 0; j < image.cols; j++){
            cv::Vec3b intensity = image.at<cv::Vec3b> (i,j);
            img.set_pixel_val_ch(i,j , 0, intensity.val[0]);
            img.set_pixel_val_ch(i,j , 1, intensity.val[1]);
            img.set_pixel_val_ch(i,j , 2, intensity.val[2]);
        }
    }
    cv::imshow("---", image);
    cv::waitKey(0);
    std::cout << "Number of channels: " << image.channels() << '\n';
    return img;
}

Image gray_scale(Image &img) {  
    if (img.depth == 3) {
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.columns; j++) {
                int val = static_cast<int>(
                    0.3 * static_cast<double>(img.get_pixel_val_ch(i, j, 0)) +
                    0.59 * static_cast<double>(img.get_pixel_val_ch(i, j, 1)) +
                    0.11 * static_cast<double>(img.get_pixel_val_ch(i, j, 2))
                );
                img.set_pixel_val(i, j, val);
            }
        }
    }
    return img;  
}

// Display functions
void display_gray(Image &img) {
    cv::Mat img_g(img.rows, img.columns, CV_8UC1);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.columns; ++j) {
            img_g.at<uchar>(i, j) = static_cast<uchar>(img.get_pixel_val(i, j)); 
        }
    }

    cv::imshow("Grayscale Image", img_g);
    cv::waitKey(0); 
}

void display_RGB(Image &img) {
    if (img.depth != 3) {
        std::cerr << "Error: Image is not in RGB format." << std::endl;
        return;
    }

    cv::Mat img_rgb(img.rows, img.columns, CV_8UC3);

    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.columns; ++j) {
            img_rgb.at<cv::Vec3b>(i, j)[0] = static_cast<uchar>(img.get_pixel_val_ch(i, j, 2)); 
            img_rgb.at<cv::Vec3b>(i, j)[1] = static_cast<uchar>(img.get_pixel_val_ch(i, j, 1)); 
            img_rgb.at<cv::Vec3b>(i, j)[2] = static_cast<uchar>(img.get_pixel_val_ch(i, j, 0));
        }
    }

    cv::imshow("RGB Image", img_rgb);
    cv::waitKey(0);
}
// Flips
void HFlip(Image &img) {
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.columns / 2; ++j) { 
            for (int k = 0; k < img.depth; ++k) {
                std::swap(img.pixel_val_ch[i][j][k], img.pixel_val_ch[i][img.columns - j - 1][k]);
            }
        }
    }
    display_RGB(img);
}

void HFlip(Image &img) {
    for (int i = 0; i < img.rows; ++i) {
        for (int j = 0; j < img.columns / 2; ++j) { 
            for (int k = 0; k < img.depth; ++k) {
                std::swap(img.pixel_val_ch[i][j][k], img.pixel_val_ch[img.rows - i - 1][j][k]);
            }
        }
    }
    display_RGB(img);
}

// Rotations
void Rrotate(Image &img){
    Image new_image = Image(img.columns, img.rows, img.depth);
    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.columns; ++j){
            new_image.set_pixel_val_ch(img.columns - j - 1, i , 0, img.get_pixel_val_ch(i, j, 0));
            new_image.set_pixel_val_ch(img.columns - j - 1, i , 1, img.get_pixel_val_ch(i, j, 1));
            new_image.set_pixel_val_ch(img.columns - j - 1, i , 2, img.get_pixel_val_ch(i, j, 2));
        }
    }
    display_RGB(img);
    display_RGB(new_image);
}

void Lrotate(Image &img){
    Image new_image = Image(img.columns, img.rows, img.depth);
    for(int i = 0; i < img.rows; ++i){
        for(int j =0; j < img.columns; ++j){
            new_image.set_pixel_val_ch(j, img.rows - i - 1, 0, img.get_pixel_val_ch(i, j, 0));
            new_image.set_pixel_val_ch(j, img.rows - i - 1 , 0, img.get_pixel_val_ch(i, j, 0));
            new_image.set_pixel_val_ch(j, img.rows - i - 1 , 0, img.get_pixel_val_ch(i, j, 0));
        }
    }
}

// Binarization

Image threshold_val (int threshold, Image &img){
    int pixel = 0, val = 0;
    Image new_image (img.rows, img.columns, img.depth);

    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.columns; ++j){
            pixel = img.get_pixel_val(i, j);
            val = pixel >= threshold ? 255 : 0;
            new_image.set_pixel_val(i, j, val);
        }
    }
    display_gray(img);
    display_gray(new_image);
    return new_image;
}

Image otsu_binarize(Image &img){
    float sum_b = 0;
    float w_B = 0;
    float w_F = 0;
    float var_max = 0;
    
    int threshold;

    std::vector<float> histogram (255, 0.0f);
    float sum = 0;

    for(int i = 0; i < 255; ++i){
        histogram[i] = 0;
    }

    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.columns; ++j){
            int pixel = img.get_pixel_val(i, j);
            histogram[pixel]++;
        }
    }

    for(int i = 0; i < 255; ++i){
        sum += histogram[i] * static_cast<float> (i);
    }

    for(int t = 0; t < 255; t++){
        w_B += histogram[t];
        if(w_B == 0) continue;

        w_F = (img.rows * img.columns) - w_B;
        if(w_F == 0) break;

        sum_b += static_cast<float> (t * histogram[t]);
        
        float m_B = sum_b/w_B;
        float m_F = (static_cast<float>(sum - sum_b)) / w_F;

        float var_between = w_B * w_F * (m_B * m_F) * (m_B - m_F);

        if(var_between > var_max){
            var_max = var_max;
            threshold = t;
        }
    }
    std::cout << "Otsu threshold value :" << threshold << std::endl;
    return threshold_val(threshold, img);
}

// Binary operations
void logicAND(Image &img_1, Image &img_2){
    if(img_1.depth == img_2.depth || img_1.rows != img_2.rows || img_1.columns != img_2.columns){
        std::cout << "Images must have same depth, rows and columns \n";
        exit(1);
    }
    int pixel = 0, val = 0;
    Image new_img_1 = otsu_binarize(img_1);
    Image new_img_2 = otsu_binarize(img_2);
    Image new_img = Image(img_1.rows, img_1.columns, img_1.depth);

    for(int i = 0; i < img_1.rows; i++){
        for(int j = 0; j < img_1.columns; j++){
            pixel = new_img_1.get_pixel_val(i,j) && new_img_2.get_pixel_val(i, j);
            val = (pixel == 1) ? 255 : 0;
            new_img.set_pixel_val(i, j, val);
        }
    }
    display_gray(img_1);
    display_gray(img_2);
    display_gray(new_img);
}

void logicNAND(Image &img_1, Image &img_2){
    if(img_1.depth == img_2.depth || img_1.rows != img_2.rows || img_1.columns != img_2.columns){
        std::cout << "Images must have same depth, rows and columns \n";
        exit(1);
    }
    int pixel = 0, val = 0;
    Image new_img_1 = otsu_binarize(img_1);
    Image new_img_2 = otsu_binarize(img_2);
    Image new_img = Image(img_1.rows, img_1.columns, img_1.depth);

    for(int i = 0; i < img_1.rows; i++){
        for(int j = 0; j < img_1.columns; j++){
            pixel = !(new_img_1.get_pixel_val(i,j) && new_img_2.get_pixel_val(i, j));
            val = (pixel == 1) ? 255 : 0;
            new_img.set_pixel_val(i, j, val);
        }
    }
    display_gray(img_1);
    display_gray(img_2);
    display_gray(new_img);
}

void logicOR(Image &img_1, Image &img_2){
    if(img_1.depth == img_2.depth || img_1.rows != img_2.rows || img_1.columns != img_2.columns){
        std::cout << "Images must have same depth, rows and columns \n";
        exit(1);
    }
    int pixel = 0, val = 0;
    Image new_img_1 = otsu_binarize(img_1);
    Image new_img_2 = otsu_binarize(img_2);
    Image new_img = Image(img_1.rows, img_1.columns, img_1.depth);

    for(int i = 0; i < img_1.rows; i++){
        for(int j = 0; j < img_1.columns; j++){
            pixel = new_img_1.get_pixel_val(i,j) || new_img_2.get_pixel_val(i, j);
            val = (pixel == 1) ? 255 : 0;
            new_img.set_pixel_val(i, j, val);
        }
    }
    display_gray(img_1);
    display_gray(img_2);
    display_gray(new_img);
}

void logicXOR(Image &img_1, Image &img_2){
    if(img_1.depth == img_2.depth || img_1.rows != img_2.rows || img_1.columns != img_2.columns){
        std::cout << "Images must have same depth, rows and columns \n";
        exit(1);
    }
    int pixel = 0, val = 0;
    Image new_img_1 = otsu_binarize(img_1);
    Image new_img_2 = otsu_binarize(img_2);
    Image new_img = Image(img_1.rows, img_1.columns, img_1.depth);

    for(int i = 0; i < img_1.rows; i++){
        for(int j = 0; j < img_1.columns; j++){
            pixel = (new_img_1.get_pixel_val(i,j) && !(new_img_2.get_pixel_val(i, j))) || 
            (new_img_2.get_pixel_val(i,j) && !(new_img_1.get_pixel_val(i, j)));
            val = (pixel == 1) ? 255 : 0;
            new_img.set_pixel_val(i, j, val);
        }
    }
    display_gray(img_1);
    display_gray(img_2);
    display_gray(new_img);
}

void addition (Image &img_1, Image &img_2){
    if(img_1.depth == img_2.depth || img_1.rows != img_2.rows || img_1.columns != img_2.columns){
        std::cout << "Images must have same depth, rows and columns \n";
        exit(1);
    }
    int pixel = 0, val = 0;
    Image new_img_1 = otsu_binarize(img_1);
    Image new_img_2 = otsu_binarize(img_2);
    Image new_img = Image(img_1.rows, img_1.columns, img_1.depth);

    for(int i = 0; i < img_1.rows; i++){
        for(int j = 0; j < img_1.columns; j++){
            pixel = new_img_1.get_pixel_val(i, j) + new_img_2.get_pixel_val(i, j);
            val = std::min(pixel, 255);
            new_img.set_pixel_val(i, j, val);
        }
    }
    display_gray(img_1);
    display_gray(img_2);
    display_gray(new_img);
}

void addition (Image &img_1, Image &img_2){
    if(img_1.depth == img_2.depth || img_1.rows != img_2.rows || img_1.columns != img_2.columns){
        std::cout << "Images must have same depth, rows and columns \n";
        exit(1);
    }
    int pixel = 0, val = 0;
    Image new_img_1 = otsu_binarize(img_1);
    Image new_img_2 = otsu_binarize(img_2);
    Image new_img = Image(img_1.rows, img_1.columns, img_1.depth);

    for(int i = 0; i < img_1.rows; i++){
        for(int j = 0; j < img_1.columns; j++){
            pixel = new_img_1.get_pixel_val(i, j) - new_img_2.get_pixel_val(i, j);
            val = std::max(pixel, 255);
            new_img.set_pixel_val(i, j, val);
        }
    }
    display_gray(img_1);
    display_gray(img_2);
    display_gray(new_img);
}

int max_pixel (Image img){
    int max_i = 0, pixel = 0;
    int val = 0;

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.columns; j++){
            pixel = img.get_pixel_val(i,j);
            max_i = std::max(pixel, max_i);
        }
    }
    return max_i;
}

int min_pixel (Image img){
    int min_i = 255, pixel = 0;
    int val = 0;

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.columns; j++){
            pixel = img.get_pixel_val(i,j);
            min_i = std::min(pixel, min_i);
        }
    }
    return min_i;
}

// Distributions
int simple_contrast(Image img){
    int max_i = max_pixel(img);
    int min_i = min_pixel(img);
    return max_i - min_i;
}

void luminance(Image &img){
    double lum = 0.0f, sum = 0.0f;
    int total = img.columns * img.rows, pixel = 0;

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.columns; j++){
            sum += img.get_pixel_val(i,j);
        }
    }
    lum = sum/total;
    std::cout << "Luminance " << lum << '\n';
}

void linear_contrast(Image img){
    Image new_img = Image(img.rows, img.columns, img.depth);
    int min_i = min_pixel(img);
    int pixel = 0;
    int val = 0;

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.columns; j++){
            pixel = img.get_pixel_val(i, j);
            val = 255 * (pixel - min_i) / simple_contrast(img);
            new_img.set_pixel_val(i,j,val);
        }
    }
}

void linear_contrast_saturation (Image &img, int sMin, int sMax){
    if((sMin > sMax) || (sMin < min_pixel(img)) || (sMax > max_pixel(img))){
        std::cout << "Invalid saturation value " << std::endl;
        exit(1);
    }

    Image new_image = Image(img.rows, img.columns, img.depth);
    int val = 0, pixel = 0;

    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.columns; ++j){
            pixel = img.get_pixel_val(i, j);
            val = 255 * (pixel - sMin) / (sMax - sMin);
            val = val > 255 ? 255 : std::max(val, 0);
            img.set_pixel_val(i, j, val); 
        }
    }
    display_RGB(img);
    display_gray(img);
}

void filtering(Image &img, double r, double g, double b){
    Image new_image = Image(img.rows, img.columns, img.depth);
    int val = 0, pixel = 0;

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.columns; j++){
            new_image.set_pixel_val_ch(i, j, 0, std::min(255, static_cast<int> (img.get_pixel_val_ch(i, j, 0) * r)));
            new_image.set_pixel_val_ch(i, j, 1, std::min(255, static_cast<int> (img.get_pixel_val_ch(i, j, 1) * g)));
            new_image.set_pixel_val_ch(i, j, 2, std::min(255, static_cast<int> (img.get_pixel_val_ch(i, j, 2) * b)));
        }
    }
}

void histogram_equalization (Image &img){
    int pixel = 0;
    int histogram[256] = {0};
    int cdf[256] = {0};
    int sk[256] = {0};
    int kk[256] = {0};

    int size = img.columns * img.rows;

    for(int i = 0; i < img.rows; ++i){
        for(int j = 0; j < img.columns; ++j){
            int pixel = img.get_pixel_val(i, j);
            histogram[pixel]++;
        }
    }

    cdf[0] = histogram[0];
    int cdf_min = histogram[0];

    for(int i = 1; i < 256; ++i){
        cdf[i] = histogram[i] + cdf[i - 1];
        cdf_min = histogram[i] == 0 ? cdf_min : std::min(histogram[i], cdf_min);
    }

    for(int i = 0; i < 256; i++){
        sk[i] = (((cdf[i] - cdf_min))/(size-cdf_min)) + 1;
    }

    for(int i = 0; i < 256; i++){
        std::cout << i << " | " << histogram[i] << " | " << cdf[i] << " | " << sk[i] << " | " << cdf_min << " | " << std::endl;
    }
}

void brightness (Image &img, int level){
    Image new_image (img.rows, img.columns, img.depth);
    int pixel = 0, val = 0;

    for(int i = 0; i < img.rows; i++){
        for(int j = 0; j < img.columns; j++){
            new_image.set_pixel_val(i, j, std::max(0, std::min(255, img.get_pixel_val(i,j) + level)));
        }
    }

    display_gray(img);
    display_gray(new_image);
}

void scaling(Image &img, double x_scale, double y_scale){
    int dest_w = img.columns * static_cast<int> (x_scale);
    int dest_h = img.rows * static_cast<int> (y_scale);

    Image new_image = Image(dest_h, dest_w, img.depth);
    
    for(int i = 0; i < dest_h; i++){
        for(int j = 0; j < dest_w; j++){
            new_image.set_pixel_val_ch(i, j, 0,img.get_pixel_val_ch(i/y_scale, j/x_scale, 0));
            new_image.set_pixel_val_ch(i, j, 1,img.get_pixel_val_ch(i/y_scale, j/x_scale, 1));
            new_image.set_pixel_val_ch(i, j, 2,img.get_pixel_val_ch(i/y_scale, j/x_scale, 2));
        }
    }

    display_RGB (img);
    display_RGB (new_image);
}

Image convolution(Image &img, const std::vector<std::vector<double>> &kernel, int k_size, double norm) {
    int rows = img.rows;
    int columns = img.columns;
    int depth = img.depth; 
    Image new_image(rows, columns, depth);

    double sum = 0.0;
    
    int k_center = k_size / 2;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            for (int c = 0; c < depth; ++c) { 
                
                for (int m = -k_center; m <= k_center; ++m) {
                    for (int n = -k_center; n <= k_center; ++n) {
                        int row_idx = i + m;
                        int col_idx = j + n;
                        
                        if (row_idx >= 0 && row_idx < rows && col_idx >= 0 && col_idx < columns) {
                            sum += img.get_pixel_val_ch(row_idx, col_idx, c) * 
                                   kernel[m + k_center][n + k_center];
                        }
                    }
                }
                new_image.set_pixel_val_ch(i, j, sum / norm, c);
                sum = 0.0;
            }
            double val = sum / norm;
            val = val < 0 ? 0 : static_cast<double>(std::min(static_cast<int>(val), 255));
        }
    }
    
    return new_image;
}

Image convolution_2D(Image &img, const std::vector<std::vector<double>> &kernel, int k_size, double norm) {
    std::vector<std::vector<double>> kern(k_size, std::vector<double>(k_size));
    
    for(int i = 0; i < k_size; i++) {
        for(int j = 0; j < k_size; j++) {
            kern[i][j] = kernel[k_size - i - 1][k_size - j - 1];
        }
    }
    return convolution(img, kern, k_size, norm);
}

void erosion(Image &img) {
    Image bin_image = otsu_binarize(img);
    std::vector<std::vector<double>> kern = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };

    Image final = convolution(bin_image, kern, 3, 3);

    display_gray(img);
    display_gray(final);
}

void Laplacien_conv(Image &img) {
    Image bin_image = otsu_binarize(img);
    std::vector<std::vector<double>> kern = {
        {1, 1, 1},
        {1, -8, 1},
        {1, 1, 1}
    };

    Image final = convolution(bin_image, kern, 3, 1);

    display_gray(img);
    display_gray(final);
}

void sharpen(Image &img) {
    std::vector<std::vector<double>> kern = {
        {1.0/4.0, 1.0/2.0, 1.0/4.0},
        {1.0/2.0, 1, 1.0/2.0},
        {1.0/4.0, 1.0/2.0, 1.0/4.0}
    };

    Image final = convolution(img, kern, 3, 4);

    display_gray(img);
    display_gray(final);
}

void edge_detector(Image &img) {
    std::vector<std::vector<double>> kern = {
        {0, 1, 0},
        {1, -4, 1},
        {0, 1, 0}
    };

    Image final = convolution(img, kern, 3, 1);

    display_gray(img);
    display_gray(final);
}

void robert_filter(Image &img) {
    std::vector<std::vector<double>> kern_1 = {
        {-1, 0},
        {0, -1}
    };

    std::vector<std::vector<double>> kern_2 = {
        {0, 1},
        {-1, 0}
    };

    Image temp = convolution_2D(img, kern_1, 2, 1);
    Image final = convolution_2D(temp, kern_2, 2, 1);

    display_gray(temp);
    display_gray(final);
}

void gauss_filter (Image &img){
    std::vector<std::vector<double>> gauss_blur = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };

    Image final = convolution(img, gauss_blur, 3, 16);
    display_gray(img);
    display_gray(final);
}

void prewitt_filter(Image &img){
    std::vector<std::vector<double>> kern_1 = {
        {-1, -1, -1},
        {0, 0, 0},
        {1, 1, 1}
    };

    std::vector<std::vector<double>> kern_2 = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    };

    Image temp = convolution(img, kern_1, 3, 1);
    Image final = convolution(temp, kern_2, 3, 1);

    display_gray(final);
    display_gray(final);
}

void sobel_filter(Image &img){
    std::vector<std::vector<double>> kern_1 = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    std::vector<std::vector<double>> kern_2 = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    Image temp = convolution(img, kern_1, 3, 1);
    Image final = convolution(temp, kern_2, 3, 1);

    display_gray(final);
    display_gray(final);
}

void smoothing_filter (Image &img){
    std::vector<std::vector<double>> kern = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    Image final = convolution(img, kern, 3, 9);

    display_gray(img);
    display_gray(img);
}

#endif