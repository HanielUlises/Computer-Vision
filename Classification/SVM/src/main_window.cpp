#include "main_window.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMessageBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QFormLayout>
#include <opencv2/opencv.hpp>

main_window::main_window(QWidget* parent)
    : QMainWindow(parent), svm_(nullptr)
{
    QWidget* central = new QWidget();
    setCentralWidget(central);

    QGroupBox* params_box = new QGroupBox("SVM Hyperparameters");
    QFormLayout* params_layout = new QFormLayout();

    kernel_box_ = new QComboBox();
    kernel_box_->addItem("Linear", KERNEL_LINEAR);
    kernel_box_->addItem("Poly",   KERNEL_POLY);
    kernel_box_->addItem("RBF",    KERNEL_RBF);
    kernel_box_->addItem("Sigmoid",KERNEL_SIGMOID);

    c_spin_ = new QDoubleSpinBox();
    c_spin_->setRange(0.0001, 10000.0);
    c_spin_->setValue(1.0);
    c_spin_->setDecimals(4);

    gamma_spin_ = new QDoubleSpinBox();
    gamma_spin_->setRange(0.000001, 10.0);
    gamma_spin_->setValue(0.01);
    gamma_spin_->setDecimals(6);

    degree_box_ = new QSpinBox();
    degree_box_->setRange(1, 20);
    degree_box_->setValue(3);

    coef0_spin_ = new QDoubleSpinBox();
    coef0_spin_->setRange(-10, 10);
    coef0_spin_->setValue(0);

    params_layout->addRow("Kernel:",    kernel_box_);
    params_layout->addRow("C:",         c_spin_);
    params_layout->addRow("Gamma:",     gamma_spin_);
    params_layout->addRow("Degree:",    degree_box_);
    params_layout->addRow("Coef0:",     coef0_spin_);

    params_box->setLayout(params_layout);

    btn_load_train_ = new QPushButton("Load Training Images");
    btn_load_test_  = new QPushButton("Load Test Images");
    btn_train_      = new QPushButton("Train SVM");
    btn_predict_    = new QPushButton("Predict Single");
    btn_evaluate_   = new QPushButton("Evaluate");

    image_preview_ = new QLabel("No image");
    image_preview_->setFixedSize(256,256);
    image_preview_->setAlignment(Qt::AlignCenter);
    image_preview_->setStyleSheet("border: 1px solid gray;");

    output_console_ = new QTextEdit();
    output_console_->setReadOnly(true);

    QVBoxLayout* layout = new QVBoxLayout();
    layout->addWidget(params_box);
    layout->addWidget(btn_load_train_);
    layout->addWidget(btn_load_test_);
    layout->addWidget(btn_train_);
    layout->addWidget(btn_predict_);
    layout->addWidget(btn_evaluate_);
    layout->addWidget(image_preview_);
    layout->addWidget(output_console_);

    central->setLayout(layout);

    connect(btn_load_train_, &QPushButton::clicked, this, &main_window::load_training_images);
    connect(btn_load_test_,  &QPushButton::clicked, this, &main_window::load_test_images);
    connect(btn_train_,      &QPushButton::clicked, this, &main_window::train_model);
    connect(btn_predict_,    &QPushButton::clicked, this, &main_window::predict_single);
    connect(btn_evaluate_,   &QPushButton::clicked, this, &main_window::evaluate_model);
}

main_window::~main_window() {
    delete svm_;
}

void main_window::load_training_images() {
    QString dir = QFileDialog::getExistingDirectory(this, "Select Training Folder");
    if (dir.isEmpty()) return;

    QDir folder(dir);
    QStringList files = folder.entryList({"*.jpg","*.png"}, QDir::Files);

    train_images_.clear();
    train_labels_.clear();

    for (QString f : files) {
        QString path = dir + "/" + f;
        cv::Mat img = cv::imread(path.toStdString());
        if (img.empty()) continue;

        train_images_.push_back(img);

        if (f.startsWith("cat_"))
            train_labels_.push_back(0);
        else if (f.startsWith("dog_"))
            train_labels_.push_back(1);
        else
            train_labels_.push_back(0);
    }

    output_console_->append("Loaded training images: " + QString::number(train_images_.size()));
}

void main_window::load_test_images() {
    QString dir = QFileDialog::getExistingDirectory(this, "Select Test Folder");
    if (dir.isEmpty()) return;

    QDir folder(dir);
    QStringList files = folder.entryList({"*.jpg","*.png"}, QDir::Files);

    test_images_.clear();
    test_labels_.clear();

    for (QString f : files) {
        QString path = dir + "/" + f;
        cv::Mat img = cv::imread(path.toStdString());
        if (img.empty()) continue;

        test_images_.push_back(img);

        if (f.startsWith("cat_"))
            test_labels_.push_back(0);
        else if (f.startsWith("dog_"))
            test_labels_.push_back(1);
        else
            test_labels_.push_back(0);
    }

    output_console_->append("Loaded test images: " + QString::number(test_images_.size()));
}


void main_window::train_model() {
    if (train_images_.empty()) {
        QMessageBox::warning(this, "Error", "No training images loaded.");
        return;
    }

    svm_parameters params;
    params.kernel = (kernel_type) kernel_box_->currentData().toInt();
    params.C = c_spin_->value();
    params.gamma = gamma_spin_->value();
    params.degree = degree_box_->value();
    params.coef0 = coef0_spin_->value();

    delete svm_;
    svm_ = new svm_classifier(params);

    svm_->train(train_images_, train_labels_);

    output_console_->append("Training completed.");
}



void main_window::evaluate_model() {
    if (!svm_) return;
    if (test_images_.empty()) return;

    auto result = svm_->evaluate(test_images_, test_labels_);

    output_console_->append("Accuracy: " + QString::number(result["accuracy"]));
}

void main_window::predict_single() {
    if (!svm_) return;

    QString file = QFileDialog::getOpenFileName(this, "Select image", "", "*.png *.jpg");
    if (file.isEmpty()) return;

    cv::Mat img = cv::imread(file.toStdString());
    if (img.empty()) return;

    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    QImage qimg(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    image_preview_->setPixmap(QPixmap::fromImage(qimg).scaled(256,256, Qt::KeepAspectRatio));

    int pred = svm_->predict(img);

    output_console_->append("Prediction: " + QString::number(pred));
}