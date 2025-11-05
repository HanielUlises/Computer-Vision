#pragma once

#include "blob.hpp"

#include <opencv4/opencv2/opencv.hpp>

#include <vector>
#include <string>

class Track{
    public:
        Track();
        ~Track();
        unsigned long int time;

        float find_distance(int, int, int, int);
        unsigned int blob_count;
        std::vector<Blob> blob_list;

        void process(std::vector<cv::Rect> rects, float intersect_ratio);
        void delete_old();
};