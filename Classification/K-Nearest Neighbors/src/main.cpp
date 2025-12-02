#include <QApplication>
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QLabel>
#include <QMessageBox>
#include <opencv2/opencv.hpp>
#include "KNN.hpp"

class main_window : public QWidget {
public:
    main_window(QWidget *parent = nullptr) : QWidget(parent), knn_(3) {
        setup_ui();
    }

    ~main_window() {}

private:
    QLabel *image_label_;
    QPushButton *upload_button_;
    QPushButton *classify_button_;
    knn knn_;
    cv::Mat current_image_;

    void setup_ui() {
        QVBoxLayout *layout = new QVBoxLayout(this);

        image_label_ = new QLabel("Image will appear here");
        image_label_->setAlignment(Qt::AlignCenter);
        layout->addWidget(image_label_);

        upload_button_ = new QPushButton("Upload Image");
        connect(upload_button_, &QPushButton::clicked, this, &main_window::upload_image);
        layout->addWidget(upload_button_);

        classify_button_ = new QPushButton("Classify");
        connect(classify_button_, &QPushButton::clicked, this, &main_window::classify_image);
        layout->addWidget(classify_button_);

        setLayout(layout);
        resize(800, 600);
    }

    void upload_image() {
        QString file_name = QFileDialog::getOpenFileName(this, "Open Image", "", "Image Files (*.bmp *.png *.jpg *.jpeg)");
        if (!file_name.isEmpty()) {
            current_image_ = cv::imread(file_name.toStdString(), cv::IMREAD_GRAYSCALE);
            if (!current_image_.empty()) {
                QImage qimg(current_image_.data, current_image_.cols, current_image_.rows, current_image_.step, QImage::Format_Grayscale8);
                image_label_->setPixmap(QPixmap::fromImage(qimg).scaled(image_label_->size(), Qt::KeepAspectRatio));
            } else {
                QMessageBox::warning(this, "Error", "Could not load the image.");
            }
        }
    }

    void classify_image() {
        if (current_image_.empty()) {
            QMessageBox::warning(this, "Warning", "Please upload an image first.");
            return;
        }

        int label = knn_.predict(current_image_);
        QMessageBox::information(this, "Classification Result", QString("The image has been classified as class: %1").arg(label));
    }
};

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    main_window window;
    window.show();
    return app.exec();
}