#include "main_window.h"
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QSlider>
#include <QCheckBox>
#include <QImage>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent),
    feature_detector(1.6, 4, 3), line_detector(10.0f, 5), show_edges(false), sigma_value(1.6) {
    capture.open(0);
    if (!capture.isOpened()) {
        std::cerr << "Error: Could not open camera" << std::endl;
        exit(1);
    }

    QWidget *central_widget = new QWidget(this);
    setCentralWidget(central_widget);
    QVBoxLayout *main_layout = new QVBoxLayout(central_widget);

    image_label = new QLabel(this);
    edges_label = new QLabel(this);
    QHBoxLayout *image_layout = new QHBoxLayout;
    image_layout->addWidget(image_label);
    image_layout->addWidget(edges_label);
    main_layout->addLayout(image_layout);

    sigma_slider = new QSlider(Qt::Horizontal, this);
    sigma_slider->setRange(10, 50); 
    sigma_slider->setValue(16); 
    connect(sigma_slider, &QSlider::valueChanged, this, &MainWindow::update_sigma);
    main_layout->addWidget(new QLabel("Sigma:"));
    main_layout->addWidget(sigma_slider);

    show_edges_checkbox = new QCheckBox("Show Edges", this);
    connect(show_edges_checkbox, &QCheckBox::toggled, this, &MainWindow::toggle_edges);
    main_layout->addWidget(show_edges_checkbox);

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &MainWindow::process_frame);
    timer->start(33); 

    setWindowTitle("lane_object_detection");
    resize(1000, 600);
}

MainWindow::~MainWindow() {
    capture.release();
}

void MainWindow::process_frame() {
    cv::Mat frame;
    capture >> frame;
    if (frame.empty()) return;

    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::Mat edges = apply_edge_detection(gray);

    // Detect keypoints
    std::vector<KeyPoint> keypoints;
    feature_detector.detect(show_edges ? edges : gray, keypoints);

    // Detect lies
    std::vector<Line> lines;
    line_detector.detect(keypoints, lines);

    cv::Mat output = frame.clone();
    draw_keypoints(output, keypoints);
    draw_lines(output, lines);

    cv::cvtColor(output, output, cv::COLOR_BGR2RGB);
    QImage qimg((uchar*)output.data, output.cols, output.rows, output.step, QImage::Format_RGB888);
    image_label->setPixmap(QPixmap::fromImage(qimg.scaled(image_label->size(), Qt::KeepAspectRatio)));

    if (show_edges) {
        cv::Mat edges_rgb;
        cv::cvtColor(edges, edges_rgb, cv::COLOR_GRAY2RGB);
        QImage qedges((uchar*)edges_rgb.data, edges_rgb.cols, edges_rgb.rows, edges_rgb.step, QImage::Format_RGB888);
        edges_label->setPixmap(QPixmap::fromImage(qedges.scaled(edges_label->size(), Qt::KeepAspectRatio)));
    } else {
        edges_label->clear();
    }
}

cv::Mat MainWindow::apply_edge_detection(const cv::Mat& gray) {
    cv::Mat edges = cv::Mat::zeros(gray.size(), CV_32F);

    for (int y = 1; y < gray.rows - 1; ++y) {
        for (int x = 1; x < gray.cols - 1; ++x) {
            float gx = gray.at<uchar>(y, x + 1) - gray.at<uchar>(y, x - 1);
            float gy = gray.at<uchar>(y + 1, x) - gray.at<uchar>(y - 1, x);
            edges.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
        }
    }

    double max_val;
    cv::minMaxLoc(edges, nullptr, &max_val);
    if (max_val > 0) edges *= 255.0 / max_val;
    edges.convertTo(edges, CV_8U);
    return edges;
}

void MainWindow::draw_keypoints(cv::Mat& image, const std::vector<KeyPoint>& keypoints) {
    for (const auto& kp : keypoints) {
        cv::circle(image, kp.pt, 3, cv::Scalar(0, 255, 0), 1);
    }
}

void MainWindow::draw_lines(cv::Mat& image, const std::vector<Line>& lines) {
    for (const auto& line : lines) {
        cv::line(image, line.start, line.end, cv::Scalar(0, 0, 255), 2);
    }
}

void MainWindow::update_sigma(int value) {
    sigma_value = value / 10.0f;
    feature_detector = FeatureDetector(sigma_value, 4, 3);
}

void MainWindow::toggle_edges(bool checked) {
    show_edges = checked;
}