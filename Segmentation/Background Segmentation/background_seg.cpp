#include <iostream>
#include <string>
#include <vector>
#include <filesystem>

#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/imgproc.hpp>
#include <opencv4/opencv2/highgui.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/cudabgsegm.hpp>
#include <opencv4/opencv2/video/background_segm.hpp>

void read_directory(const std::filesystem::path& directory, std::vector<std::string>& files) {
    files.clear();
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        files.emplace_back(entry.path().filename().string());
    }
}

int main() {
    std::string path {};
    std::vector<std::string> file_list;
    read_directory(path, file_list);
    std::sort(file_list.begin(), file_list.end());

    cv::Mat frame, fgmask;
    cv::cuda::GpuMat d_frame, d_fgmask;

    cv::Ptr<cv::BackgroundSubtractor> mog = cv::cuda::createBackgroundSubtractorMOG();
    cv::Ptr<cv::BackgroundSubtractor> mog2 = cv::cuda::createBackgroundSubtractorMOG2();


    std::chrono::steady_clock::time_point begin, end;

    for(int i = 1; i < static_cast<int>(file_list.size()); i++){
        begin = std::chrono::steady_clock::now();

        std::string filename = file_list[i];
        std::string prev_filename = file_list[i - 1];
        if(std::string::npos == filename.find(".jpg") || std::string::npos == prev_filename.find(".jpg"))
            continue;
        
        frame = cv::imread(path + filename);
        d_frame.upload(frame);
        mog -> apply(d_frame, d_fgmask, 0.01);
        d_fgmask.download(fgmask);
        
        cv::imshow("frame", frame);
        cv::imshow("fgmask", fgmask);

        end = std::chrono::steady_clock::now();
        std::cout << "Res: " << frame.size().width << "x" << frame.size().height << std::endl;

        int ch = cv::waitKey(38);
        if('q' == static_cast<char>(ch))
            break;

    }   
}