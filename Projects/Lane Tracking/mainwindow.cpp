#include "mainwindow.h"
#include <QLabel>
#include <QVBoxLayout>
#include <QImage>
#include <QTimer>
#include <opencv2/imgproc.hpp>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    capture.open(0);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        exit(1);
    }

    sift = cv::SIFT::create();

    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    imageLabel = new QLabel(this);
    layout->addWidget(imageLabel);

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::processFrame);
    timer->start(33); 

    setWindowTitle("Lane Object Detection with SIFT");
    resize(800, 600);
}

MainWindow::~MainWindow() {
    capture.release();
}

void MainWindow::processFrame() {
    cv::Mat frame;
    capture >> frame;
    if (frame.empty()) {
        std::cerr << "Error: Empty frame" << std::endl;
        return;
    }

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    sift->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);

    cv::Mat output;
    cv::drawKeypoints(frame, keypoints, output, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::cvtColor(output, output, cv::COLOR_BGR2RGB); 
    QImage qimg((uchar*)output.data, output.cols, output.rows, output.step, QImage::Format_RGB888);
    imageLabel->setPixmap(QPixmap::fromImage(qimg.scaled(imageLabel->size(), Qt::KeepAspectRatio)));
}