#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QLabel>
#include <QMessageBox>
#include "SVM.h"

class MainWindow : public QWidget {
public:
    MainWindow(QWidget *parent = nullptr) : QWidget(parent), svm(SVMParameters{RBF, 1.0, 0.5, 3, 0.0}) {
        setupUI();
        svm.setLogging(true);  // Enable logging
    }

private:
    QLabel *imageLabel;
    QPushButton *uploadButton;
    QPushButton *classifyButton;
    SVM svm;
    std::vector<cv::Mat> trainImages;
    std::vector<int> trainLabels;
    cv::Mat currentImage;

    void setupUI() {
        QVBoxLayout *layout = new QVBoxLayout(this);

        imageLabel = new QLabel("Image will appear here");
        imageLabel->setAlignment(Qt::AlignCenter);
        layout->addWidget(imageLabel);

        uploadButton = new QPushButton("Upload Image");
        connect(uploadButton, &QPushButton::clicked, this, &MainWindow::uploadImage);
        layout->addWidget(uploadButton);

        classifyButton = new QPushButton("Classify");
        connect(classifyButton, &QPushButton::clicked, this, &MainWindow::classifyImage);
        layout->addWidget(classifyButton);

        setLayout(layout);
        resize(800, 600);

        loadTrainingData();  // Load some training data
        svm.train(trainImages, trainLabels);
        svm.displaySummary();  // Display the model summary
    }

    void loadTrainingData() {
        // This function should load actual training data
        trainImages = {};
        trainLabels = {};
    }

    void uploadImage() {
        QString fileName = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.bmp *.png *.jpg *.jpeg)");
        if (!fileName.isEmpty()) {
            currentImage = cv::imread(fileName.toStdString(), cv::IMREAD_GRAYSCALE);
            if (!currentImage.empty()) {
                QImage qimg(currentImage.data, currentImage.cols, currentImage.rows, currentImage.step, QImage::Format_Grayscale8);
                imageLabel->setPixmap(QPixmap::fromImage(qimg).scaled(imageLabel->size(), Qt::KeepAspectRatio));
            } else {
                QMessageBox::warning(this, "Error", "Could not load the image.");
            }
        }
    }

    void classifyImage() {
        if (currentImage.empty()) {
            QMessageBox::warning(this, "Warning", "Please upload an image first.");
            return;
        }

        int label = svm.predict(currentImage);
        QMessageBox::information(this, "Classification Result", QString("The image has been classified as class: %1").arg(label));
    }
};

int main
