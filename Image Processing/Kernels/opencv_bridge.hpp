#pragma once
#include <opencv2/core.hpp>

template<typename T>
struct CvView {
    const T* src;
    T* dst;
    int width;
    int height;
    int stride;
};

template<typename T>
inline CvView<T> view(cv::Mat& src, cv::Mat& dst) {
    return {
        src.ptr<T>(),
        dst.ptr<T>(),
        src.cols,
        src.rows,
        static_cast<int>(src.step1())
    };
}
