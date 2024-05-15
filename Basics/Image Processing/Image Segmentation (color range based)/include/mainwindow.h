#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>
#include <QLineEdit>

class MainWindow : public QWidget {
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow() = default;

private slots:
    void onLoadImage();
    void onFilterImage();
    void onSaveImage();

private:
    QLabel *originalImageLabel;
    QLabel *filteredImageLabel;
    QPushButton *loadButton;
    QPushButton *filterButton;
    QPushButton *saveButton;
    QImage loadedImage;
    QImage filteredImage;
    QLineEdit *minRed, *maxRed;
    QLineEdit *minGreen, *maxGreen;
    QLineEdit *minBlue, *maxBlue;

    void updateImageDisplay();
};

#endif // MAINWINDOW_H
