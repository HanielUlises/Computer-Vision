#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

class QLabel;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void processFrame();

private:
    QLabel *imageLabel;
    QTimer *timer;
    cv::VideoCapture capture;
    cv::Ptr<cv::SIFT> sift;
};

#endif // MAINWINDOW_H