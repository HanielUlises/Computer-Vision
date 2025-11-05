#pragma once

#include <opencv4/opencv2/opencv.hpp>

class Blob{
        unsigned long int create_time;
        unsigned long int update_time;

    public:
        Blob(int, unsigned long int);
        int id;
        cv::Rect rect;

        void set_update_time(unsigned long int _update_time);
        bool is_removed(unsigned long int time);
        unsigned long int get_life();
};