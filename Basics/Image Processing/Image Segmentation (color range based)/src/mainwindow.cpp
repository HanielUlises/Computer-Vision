#include "mainwindow.h"
#include "mainmenu.h"
#include <QColor>
#include <QMessageBox>
#include <QIntValidator>

MainWindow::MainWindow(QWidget *parent) : QWidget(parent) {
    originalImageLabel = new QLabel(this);
    filteredImageLabel = new QLabel(this);

    int imageWidth = 400;
    int imageHeight = 300;
    originalImageLabel->setFixedSize(imageWidth, imageHeight);
    filteredImageLabel->setFixedSize(imageWidth, imageHeight);

    setPlaceholderImages();

    loadButton = new QPushButton("Load Image", this);
    filterButton = new QPushButton("Filter Image", this);
    saveButton = new QPushButton("Save Filtered Image", this);
    backButton = new QPushButton("← Back to Main Menu", this);

    setupSliders(minRedSlider, maxRedSlider);
    setupSliders(minGreenSlider, maxGreenSlider);
    setupSliders(minBlueSlider, maxBlueSlider);

    // Layout for color range input
    QVBoxLayout *colorLayout = new QVBoxLayout;
    colorLayout->addWidget(new QLabel("Min Red:"));
    colorLayout->addWidget(minRedSlider);
    colorLayout->addWidget(new QLabel("Max Red:"));
    colorLayout->addWidget(maxRedSlider);
    colorLayout->addWidget(new QLabel("Min Green:"));
    colorLayout->addWidget(minGreenSlider);
    colorLayout->addWidget(new QLabel("Max Green:"));
    colorLayout->addWidget(maxGreenSlider);
    colorLayout->addWidget(new QLabel("Min Blue:"));
    colorLayout->addWidget(minBlueSlider);
    colorLayout->addWidget(new QLabel("Max Blue:"));
    colorLayout->addWidget(maxBlueSlider);

    QHBoxLayout *hLayout = new QHBoxLayout;
    hLayout->addWidget(originalImageLabel);
    hLayout->addLayout(colorLayout);
    hLayout->addWidget(filteredImageLabel);

    QVBoxLayout *vLayout = new QVBoxLayout(this);
    vLayout->addLayout(hLayout);
    vLayout->addWidget(loadButton);
    vLayout->addWidget(filterButton);
    vLayout->addWidget(saveButton);
    vLayout->addWidget(backButton);

    connect(loadButton, &QPushButton::clicked, this, &MainWindow::onLoadImage);
    connect(filterButton, &QPushButton::clicked, this, &MainWindow::onFilterImage);
    connect(saveButton, &QPushButton::clicked, this, &MainWindow::onSaveImage);
    connect(backButton, &QPushButton::clicked, this, &MainWindow::onBackToMainMenu);
}

void MainWindow::setupSliders(QSlider*& minSlider, QSlider*& maxSlider) {
    minSlider = new QSlider(Qt::Horizontal, this);
    maxSlider = new QSlider(Qt::Horizontal, this);
    
    minSlider->setRange(0, 255);
    maxSlider->setRange(0, 255);

    minSlider->setValue(0);
    maxSlider->setValue(255);
}

void MainWindow::onLoadImage() {
    QString filePath = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.png *.jpg *.bmp)");
    if (!filePath.isEmpty()) {
        if (!loadedImage.load(filePath)) {
            QMessageBox::warning(this, "Error", "Failed to load the image.");
            return;
        }
        updateImageDisplay();
    }
}

void MainWindow::onFilterImage() {
    int rMin = minRedSlider->value();
    int rMax = maxRedSlider->value();
    int gMin = minGreenSlider->value();
    int gMax = maxGreenSlider->value();
    int bMin = minBlueSlider->value();
    int bMax = maxBlueSlider->value();

    filteredImage = loadedImage;

    for (int y = 0; y < filteredImage.height(); y++) {
        for (int x = 0; x < filteredImage.width(); x++) {
            QColor color(filteredImage.pixel(x, y));
            int red = color.red();
            int green = color.green();
            int blue = color.blue();

            // Only keep the pixel if it falls within all specified ranges
            if (red >= rMin && red <= rMax && green >= gMin && green <= gMax && blue >= bMin && blue <= bMax) {
                // Pixel is within range, keep it as is
                filteredImage.setPixel(x, y, qRgb(red, green, blue));
            } else {
                // Pixel is out of range, set it to black
                filteredImage.setPixel(x, y, qRgb(0, 0, 0));
            }
        }
    }

    updateImageDisplay();
}

void MainWindow::onSaveImage() {
    QString filePath = QFileDialog::getSaveFileName(this, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp)");
    if (!filePath.isEmpty()) {
        if (!filteredImage.save(filePath)) {
            QMessageBox::warning(this, "Error", "Failed to save the image.");
        }
    }
}

void MainWindow::onBackToMainMenu() {
    MainMenu *mainMenu = new MainMenu();
    mainMenu->show();
    this->close();
}

void MainWindow::updateImageDisplay() {
    originalImageLabel->setPixmap(QPixmap::fromImage(loadedImage).scaled(originalImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    filteredImageLabel->setPixmap(QPixmap::fromImage(filteredImage).scaled(filteredImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}

void MainWindow::setPlaceholderImages() {
    int imageWidth = originalImageLabel->width();
    int imageHeight = originalImageLabel->height();

    QPixmap placeholder(imageWidth, imageHeight);
    placeholder.fill(Qt::gray);

    originalImageLabel->setPixmap(placeholder);
    filteredImageLabel->setPixmap(placeholder);
}
