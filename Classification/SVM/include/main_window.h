#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QSpinBox>

#include <vector>
#include <opencv2/opencv.hpp>

#include "SVM.h"  

class main_window : public QMainWindow
{
    Q_OBJECT

public:
    explicit main_window(QWidget* parent = nullptr);
    ~main_window();

private slots:
    void load_training_images();
    void load_test_images();
    void train_model();
    void evaluate_model();
    void predict_single();

private:

    // Hyperparameter controls
    QComboBox*      kernel_box_;
    QDoubleSpinBox* c_spin_;
    QDoubleSpinBox* gamma_spin_;
    QSpinBox*       degree_box_;
    QDoubleSpinBox* coef0_spin_;

    // Action buttons
    QPushButton* btn_load_train_;
    QPushButton* btn_load_test_;
    QPushButton* btn_train_;
    QPushButton* btn_predict_;
    QPushButton* btn_evaluate_;

    // Visual output
    QLabel* image_preview_;
    QTextEdit* output_console_;


    std::vector<cv::Mat> train_images_;
    std::vector<int> train_labels_;

    std::vector<cv::Mat> test_images_;
    std::vector<int> test_labels_;

    svm_classifier* svm_;
};

#endif 
