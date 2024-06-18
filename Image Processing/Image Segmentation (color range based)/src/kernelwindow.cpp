#include "kernelwindow.h"
#include "mainmenu.h"
#include <QColor>
#include <QMessageBox>
#include <cmath>

KernelWindow::KernelWindow(QWidget *parent) : QWidget(parent) {
    originalImageLabel = new QLabel(this);
    processedImageLabel = new QLabel(this);

    int imageWidth = 400;
    int imageHeight = 300;
    originalImageLabel->setFixedSize(imageWidth, imageHeight);
    processedImageLabel->setFixedSize(imageWidth, imageHeight);

    setPlaceholderImages();

    loadButton = new QPushButton("Load Image", this);
    kernelComboBox = new QComboBox(this);
    kernelComboBox->addItem("Sobel");
    kernelComboBox->addItem("Prewitt");
    kernelComboBox->addItem("Laplacian");
    applyKernelButton = new QPushButton("Apply Kernel", this);
    saveButton = new QPushButton("Save Processed Image", this);
    backButton = new QPushButton("← Back to Main Menu", this);

    QHBoxLayout *hLayout = new QHBoxLayout;
    hLayout->addWidget(originalImageLabel);
    hLayout->addWidget(processedImageLabel);

    QVBoxLayout *vLayout = new QVBoxLayout(this);
    vLayout->addLayout(hLayout);
    vLayout->addWidget(loadButton);
    vLayout->addWidget(kernelComboBox);
    vLayout->addWidget(applyKernelButton);
    vLayout->addWidget(saveButton);
    vLayout->addWidget(backButton);

    connect(loadButton, &QPushButton::clicked, this, &KernelWindow::onLoadImage);
    connect(applyKernelButton, &QPushButton::clicked, this, &KernelWindow::onApplyKernel);
    connect(saveButton, &QPushButton::clicked, this, &KernelWindow::onSaveImage);
    connect(backButton, &QPushButton::clicked, this, &KernelWindow::onBackToMainMenu);
}

void KernelWindow::onLoadImage() {
    QString filePath = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.png *.jpg *.bmp)");
    if (!filePath.isEmpty()) {
        if (!loadedImage.load(filePath)) {
            QMessageBox::warning(this, "Error", "Failed to load the image.");
            return;
        }
        processedImage = QImage();
        updateImageDisplay();
    }
}

void KernelWindow::onApplyKernel() {
    if (loadedImage.isNull()) {
        QMessageBox::warning(this, "Error", "Please load an image first.");
        return;
    }

    QString selectedKernel = kernelComboBox->currentText();

    if (selectedKernel == "Sobel") {
        applySobelFilter();
    } else if (selectedKernel == "Prewitt") {
        applyPrewittFilter();
    } else if (selectedKernel == "Laplacian") {
        applyLaplacianFilter();
    }

    updateImageDisplay();
}

void KernelWindow::onSaveImage() {
    QString filePath = QFileDialog::getSaveFileName(this, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg);;BMP Files (*.bmp)");
    if (!filePath.isEmpty()) {
        if (!processedImage.save(filePath)) {
            QMessageBox::warning(this, "Error", "Failed to save the image.");
        }
    }
}

void KernelWindow::onBackToMainMenu() {
    MainMenu *mainMenu = new MainMenu();
    mainMenu->show();
    this->close();
}

void KernelWindow::updateImageDisplay() {
    originalImageLabel->setPixmap(QPixmap::fromImage(loadedImage).scaled(originalImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    if (!processedImage.isNull()) {
        processedImageLabel->setPixmap(QPixmap::fromImage(processedImage).scaled(processedImageLabel->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
    } else {
        processedImageLabel->clear();
    }
}

void KernelWindow::setPlaceholderImages() {
    int imageWidth = originalImageLabel->width();
    int imageHeight = originalImageLabel->height();

    QPixmap placeholder(imageWidth, imageHeight);
    placeholder.fill(Qt::gray);

    originalImageLabel->setPixmap(placeholder);
    processedImageLabel->setPixmap(placeholder);
}

void KernelWindow::applySobelFilter() {
    // Sobel kernels for edge detection
    const int kernelSize = 3;
    int gx[kernelSize][kernelSize] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    int gy[kernelSize][kernelSize] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    int width = loadedImage.width();
    int height = loadedImage.height();
    processedImage = QImage(width, height, QImage::Format_Grayscale8);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int sumX = 0;
            int sumY = 0;

            // Apply kernels to the neighborhood
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixelValue = qGray(loadedImage.pixel(x + kx, y + ky));
                    sumX += pixelValue * gx[ky + 1][kx + 1];
                    sumY += pixelValue * gy[ky + 1][kx + 1];
                }
            }

            // Calculate the magnitude of the gradient
            int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            magnitude = qBound(0, magnitude, 255);

            // Set the pixel value in the processed image
            processedImage.setPixel(x, y, qRgb(magnitude, magnitude, magnitude));
        }
    }
}

void KernelWindow::applyPrewittFilter() {
    // Prewitt kernels for edge detection
    const int kernelSize = 3;
    int gx[kernelSize][kernelSize] = {
        {-1, 0, 1},
        {-1, 0, 1},
        {-1, 0, 1}
    };
    int gy[kernelSize][kernelSize] = {
        {-1, -1, -1},
        {0, 0, 0},
        {1, 1, 1}
    };

    int width = loadedImage.width();
    int height = loadedImage.height();
    processedImage = QImage(width, height, QImage::Format_Grayscale8);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int sumX = 0;
            int sumY = 0;

            // Applying kernels to the neighborhood
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixelValue = qGray(loadedImage.pixel(x + kx, y + ky));
                    sumX += pixelValue * gx[ky + 1][kx + 1];
                    sumY += pixelValue * gy[ky + 1][kx + 1];
                }
            }

            // Magnitude of the gradient
            int magnitude = static_cast<int>(std::sqrt(sumX * sumX + sumY * sumY));
            magnitude = qBound(0, magnitude, 255);

            processedImage.setPixel(x, y, qRgb(magnitude, magnitude, magnitude));
        }
    }
}

void KernelWindow::applyLaplacianFilter() {
    // Laplacian kernel for edge detection
    const int kernelSize = 3;
    int kernel[kernelSize][kernelSize] = {
        {0, -1, 0},
        {-1, 4, -1},
        {0, -1, 0}
    };

    int width = loadedImage.width();
    int height = loadedImage.height();
    processedImage = QImage(width, height, QImage::Format_Grayscale8);

    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int sum = 0;

            // Apply kernel to the neighborhood
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int pixelValue = qGray(loadedImage.pixel(x + kx, y + ky));
                    sum += pixelValue * kernel[ky + 1][kx + 1];
                }
            }

            // Final value
            int value = qBound(0, sum, 255);

            // Pixel value in the processed image
            processedImage.setPixel(x, y, qRgb(value, value, value));
        }
    }
}
