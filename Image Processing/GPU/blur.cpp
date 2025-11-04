#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace chrono;

int main() {
    cout << "OpenCV version: " << CV_VERSION << endl;
    cout << "CUDA devices: " << cuda::getCudaEnabledDeviceCount() << endl;
    if (cuda::getCudaEnabledDeviceCount() == 0) {
        cerr << "No CUDA device found!" << endl;
        return -1;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Cannot open camera!" << endl;
        return -1;
    }

    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    cuda::GpuMat gpu_frame, gpu_gray, gpu_blur, gpu_canny;

    auto blur_filter = cuda::createGaussianFilter(CV_8UC3, CV_8UC3, Size(15, 15), 0);
    auto canny_detector = cuda::createCannyEdgeDetector(50, 150);

    Mat frame;
    auto start = high_resolution_clock::now();
    int frame_count = 0;

    cout << "Press 'q' to quit.\n";

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        gpu_frame.upload(frame);

        blur_filter->apply(gpu_frame, gpu_blur);
        cuda::cvtColor(gpu_blur, gpu_gray, COLOR_BGR2GRAY);
        canny_detector->detect(gpu_gray, gpu_canny);

        Mat result;
        gpu_canny.download(result);

        frame_count++;
        auto now = high_resolution_clock::now();
        double elapsed = duration_cast<milliseconds>(now - start).count() / 1000.0;
        if (elapsed > 1.0) {
            cout << "FPS: " << frame_count / elapsed << "     \r" << flush;
            frame_count = 0;
            start = now;
        }

        imshow("CUDA Canny Edge", result);
        if (waitKey(1) == 'q') break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}