#include "slam_system.hpp"
#include "visualizer.hpp"
#include <opencv2/highgui.hpp>
#include <iostream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: visual_slam <video_or_image_dir> [calib.yaml]\n"
                  << "  TUM: visual_slam /path/to/tum_sequence/\n"
                  << "  Video: visual_slam video.mp4\n";
        return 1;
    }

    vslam::CameraIntrinsics K;
    vslam::SLAMConfig cfg;

    if (argc >= 3) {
        cv::FileStorage fs(argv[2], cv::FileStorage::READ);
        double fx, fy, cx, cy;
        fs["fx"] >> fx; fs["fy"] >> fy;
        fs["cx"] >> cx; fs["cy"] >> cy;
        cv::Mat dist;
        fs["dist_coeffs"] >> dist;
        K = vslam::CameraIntrinsics(fx, fy, cx, cy, dist);
    } else {
        // TUM fr1 default calibration
        K = vslam::CameraIntrinsics(517.3, 516.5, 318.6, 255.3);
    }

    vslam::SLAMSystem slam(K, cfg);
    slam.enable_visualization(true);

    std::string input(argv[1]);
    bool is_video = (input.find(".mp4") != std::string::npos ||
                     input.find(".avi") != std::string::npos ||
                     input.find(".mkv") != std::string::npos);

    auto process_frame = [&](cv::Mat& frame, double ts) {
        if (frame.empty()) return false;
        bool ok = slam.process_image(frame, ts);

        cv::Mat disp = vslam::Visualizer::draw_tracked_points(
            slam.map()->get_active_keyframes(1).empty()
                ? std::make_shared<vslam::Frame>(-1)
                : slam.map()->get_active_keyframes(1).front()
        );
        cv::imshow("SLAM Feed", frame);
        return ok;
    };

    if (is_video) {
        cv::VideoCapture cap(input);
        if (!cap.isOpened()) {
            std::cerr << "Cannot open video: " << input << "\n";
            return 1;
        }
        cv::Mat frame;
        double fps = cap.get(cv::CAP_PROP_FPS);
        int idx = 0;
        while (cap.read(frame)) {
            slam.process_image(frame, idx / fps);
            cv::imshow("SLAM Feed", frame);
            if (cv::waitKey(1) == 'q') break;
            ++idx;
        }
    } else {
        std::vector<fs::path> paths;
        for (const auto& e : fs::directory_iterator(input)) {
            auto ext = e.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg")
                paths.push_back(e.path());
        }
        std::sort(paths.begin(), paths.end());

        for (size_t i = 0; i < paths.size(); ++i) {
            cv::Mat img = cv::imread(paths[i].string());
            if (img.empty()) continue;
            slam.process_image(img, static_cast<double>(i));
            cv::imshow("SLAM Feed", img);
            if (cv::waitKey(10) == 'q') break;
        }
    }

    slam.save_trajectory("trajectory_tum.txt");
    std::cout << "\n[Done] Keyframes: " << slam.num_keyframes()
              << " | Landmarks: " << slam.num_landmarks() << "\n"
              << "Trajectory saved to trajectory_tum.txt\n";

    return 0;
}