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

#endif