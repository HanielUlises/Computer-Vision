#ifndef KERNELWINDOW_H
#define KERNELWINDOW_H

#include <QWidget>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QImage>
#include <QPixmap>
#include <QFileDialog>

class KernelWindow : public QWidget {
    Q_OBJECT

public:
    KernelWindow(QWidget *parent = nullptr);
    ~KernelWindow() = default;

private slots:
    void onLoadImage();
    void onApplyKernel();
    void onSaveImage();
    void onBackToMainMenu();

private:
    QLabel *originalImageLabel;
    QLabel *processedImageLabel;
    QPushButton *loadButton;
    QComboBox *kernelComboBox;
    QPushButton *applyKernelButton;
    QPushButton *saveButton;
    QPushButton *backButton;
    QImage loadedImage;
    QImage processedImage;

    void updateImageDisplay();
    void setPlaceholderImages();
    void applySobelFilter();
    void applyPrewittFilter();
    void applyLaplacianFilter();
};

#endif // KERNELWINDOW_H
