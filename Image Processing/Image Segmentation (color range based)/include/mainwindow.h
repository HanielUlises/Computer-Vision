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
    void onBackToMainMenu();

private:
    QLabel *originalImageLabel;
    QLabel *filteredImageLabel;
    QPushButton *loadButton;
    QPushButton *filterButton;
    QPushButton *saveButton;
    QPushButton *backButton;
    QImage loadedImage;
    QImage filteredImage;
    QSlider *minRedSlider, *maxRedSlider;
    QSlider *minGreenSlider, *maxGreenSlider;
    QSlider *minBlueSlider, *maxBlueSlider;

    void updateImageDisplay();
    void setPlaceholderImages();
    void setupSliders(QSlider*& minSlider, QSlider*& maxSlider);
};

#endif // MAINWINDOW_H
