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

#endif