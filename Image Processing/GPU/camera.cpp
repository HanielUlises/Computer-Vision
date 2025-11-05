#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/cudafilters.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>

cv::Mat mask;
cv::Scalar lower, higher;

int h_max = 170, s_max = 110, v_max = 255;
uint64_t begin;

void read_directory(const std::filesystem::path& directory, std::vector<std::string>& files) {
    files.clear();
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        files.emplace_back(entry.path().filename().string());
    }
}

void find_contours(cv::cuda::GpuMat input, cv::cuda::GpuMat &output){
    cv::cuda::GpuMat output1_cuda, output2_cuda;

    cv::Ptr<cv::cuda::Filter> gauss_1 = cv::cuda::createGaussianFilter(input.type(), input.type(), cv::Size(25, 25), 0.8);
    cv::Ptr<cv::cuda::Filter> gauss_2 = cv::cuda::createGaussianFilter(input.type(), input.type(), cv::Size(15, 15), 0.5);

    gauss_1 -> apply(input, output1_cuda);
    gauss_2 -> apply(input, output2_cuda);
    cv::cuda::absdiff(output1_cuda, output2_cuda, output);

    cv::cuda::threshold(output, output, 10, 255, cv::THRESH_BINARY);

}

static void on_trackbar(int pos, void *){
    std::cout << "Tracker updated " << pos << std::endl;
    higher = cv::Scalar(h_max, s_max, v_max);
}



int main() {
    std::string path {};
    cv::Mat frame {};
    cv::Mat g1, g2;

    cv::cuda::GpuMat d_frame, d_out;
    cv::VideoCapture capture = cv::VideoCapture(path);

    std::chrono::steady_clock::time_point begin, end;

    while(capture.grab()){
        begin = std::chrono::steady_clock::now();
        capture.read(frame);
        
        // Process
        // cv::GaussianBlur(frame, frame, cv::Size(15, 15), 0.9);

        cv::Mat result {};
        d_frame.upload(frame);
        find_contours(d_frame, d_out);
        d_out.download(frame);

        cv::resize(result, result, cv::Size(960, 540));
        cv::imshow("contour", result);

        cv::resize(frame, frame, cv::Size(960, 540));
        cv::imshow("frame", frame);

        int ch = cv::waitKey(10);

        if('q' == static_cast<char>(ch))
            break;

        end = std::chrono::steady_clock::now();
        std::cout << "Time duration " << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << "ms" << std::endl; 

    }


    {
        cv::Mat img {}, gray {}, hsv {};

        std::vector<std::string> file_list;
        read_directory(path, file_list);
        std::sort(file_list.begin(), file_list.end());

        int low_threshold = 10, ratio = 4;

        cv::namedWindow("mask", cv::WINDOW_AUTOSIZE);
        cv::createTrackbar("h_max", "mask", &h_max, 255);
        cv::createTrackbar("s_max", "mask", &s_max, 255);
        cv::createTrackbar("v_max", "mask", &v_max, 255);

        cv::Scalar low = {0, 0, 200};
        cv::Scalar high = {170, 110, 255};

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
            cv::inRange(hsv, low, high, mask);
            cv::imshow("img", gray);
            cv::imshow("hsv", hsv);
            cv::imshow("edges", edges);

            int ch = cv::waitKey(0);
        }

        
    }
}