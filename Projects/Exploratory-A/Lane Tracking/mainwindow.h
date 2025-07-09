#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include "feature_detector.h"
#include "line_detector.h"

class QLabel;
class QSlider;
class QCheckBox;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void process_frame();
    void update_sigma(int value);
    void toggle_edges(bool checked);

private:
    QLabel *image_label;
    QLabel *edges_label;
    QTimer *timer;
    QSlider *sigma_slider;
    QCheckBox *show_edges_checkbox;
    cv::VideoCapture capture;
    FeatureDetector feature_detector;
    LineDetector line_detector;
    bool show_edges;
    float sigma_value;

    cv::Mat apply_edge_detection(const cv::Mat& gray);
    void draw_keypoints(cv::Mat& image, const std::vector<KeyPoint>& keypoints);
    void draw_lines(cv::Mat& image, const std::vector<Line>& lines);
};

#endif // MAIN_WINDOW_H