#include "mainwindow.h"
#include <QColor>

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
    originalImageLabel = new QLabel(this);
    filteredImageLabel = new QLabel(this);

    loadButton = new QPushButton("Load Image", this);
    filterButton = new QPushButton("Filter Image", this);
    saveButton = new QPushButton("Save Filtered Image", this);

    minRed = new QLineEdit("0", this);
    maxRed = new QLineEdit("255", this);
    minGreen = new QLineEdit("0", this);
    maxGreen = new QLineEdit("255", this);
    minBlue = new QLineEdit("0", this);
    maxBlue = new QLineEdit("255", this);

    QVBoxLayout *colorLayout = new QVBoxLayout;
    colorLayout->addWidget(new QLabel("Min Red:"));
    colorLayout->addWidget(minRed);
    colorLayout->addWidget(new QLabel("Max Red:"));
    colorLayout->addWidget(maxRed);
    colorLayout->addWidget(new QLabel("Min Green:"));
    colorLayout->addWidget(minGreen);
    colorLayout->addWidget(new QLabel("Max Green:"));
    colorLayout->addWidget(maxGreen);
    colorLayout->addWidget(new QLabel("Min Blue:"));
    colorLayout->addWidget(minBlue);
    colorLayout->addWidget(new QLabel("Max Blue:"));
    colorLayout->addWidget(maxBlue);

    QHBoxLayout *hLayout = new QHBoxLayout;
    hLayout->addWidget(originalImageLabel);
    hLayout->addLayout(colorLayout);
    hLayout->addWidget(filteredImageLabel);

    QVBoxLayout *vLayout = new QVBoxLayout(this);
    vLayout->addLayout(hLayout);
    vLayout->addWidget(loadButton);
    vLayout->addWidget(filterButton);
    vLayout->addWidget(saveButton);

    connect(loadButton, &QPushButton::clicked, this, &MainWindow::onLoadImage);
    connect(filterButton, &QPushButton::clicked, this, &MainWindow::onFilterImage);
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::onSaveImage);
}

void MainWindow::onLoadImage() {
    QString filePath = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.png *.jpg *.bmp)");
    if (!filePath.isEmpty()) {
        loadedImage.load(filePath);
        originalImageLabel->setPixmap(QPixmap::fromImage(loadedImage));
    }
}

void MainWindow::onFilterImage() {
    int rMin = minRed->text().toInt();
    int rMax = maxRed->text().toInt();
    int gMin = minGreen->text().toInt();
    int gMax = maxGreen->text().toInt();
    int bMin = minBlue->text().toInt();
    int bMax = maxBlue->text().toInt();

    filteredImage = loadedImage;
    for (int y = 0; y < filteredImage.height(); y++) {
        for (int x = 0; x < filteredImage.width(); x++) {
            QColor color(filteredImage.pixel(x, y));
            int red = color.red();
            int green = color.green();
            int blue = color.blue();
            if (red < rMin || red > rMax || green < gMin || green > gMax || blue < bMin || blue > bMax) {
                filteredImage.setPixel(x, y, qRgb(0, 0, 0));
            }
        }
    }

    filteredImageLabel->setPixmap(QPixmap::fromImage(filteredImage));
}

void MainWindow::onSaveImage() {
    QString filePath = QFileDialog::getSaveFileName(this, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp)");
    if (!filePath.isEmpty()) {
        filteredImage.save(filePath);
    }
}
