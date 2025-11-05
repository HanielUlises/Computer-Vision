#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>

cv::Mat mask;
cv::Scalar lower, higher;
int h_max = 170, s_max = 110, v_max = 255;

void read_directory(const std::filesystem::path& directory, std::vector<std::string>& files) {
    files.clear();
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        files.emplace_back(entry.path().filename().string());
    }
}

int main() {
    std::string path {};
    cv::Mat img {}, gray {}, hsv {};

    std::vector<std::string> file_list;
    read_directory(path, file_list);
    std::sort(file_list.begin(), file_list.end());

    cv::VideoCapture _capture = cv::VideoCapture(0);

    int low_threshold = 10, ratio = 4;
    
    while(_capture.grab()){
        _capture.read(img);
        cv::imshow("img", img);
        int ch = cv::waitKey(0);
        if('q' == static_cast<char>(ch)){
            break;
        }
    }

    for(std::string filename : file_list){
        if(std::string::npos == filename.find(".jpg"))
            continue;
        std::cout << filename << std::endl;
        img = cv::imread(path + filename);

        cv::cvtColor(img, hsv, cv::COLOR_BGR2HSV);

        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::blur(gray, gray, cv::Size(7,7));
        cv::Mat edges {}, mask {};
        cv::Canny(gray, edges, low_threshold, low_threshold * ratio);
        cv::imshow("img", gray);
        cv::imshow("hsv", hsv);
        cv::imshow("edges", edges);

        int ch = cv::waitKey(0);
    }
}