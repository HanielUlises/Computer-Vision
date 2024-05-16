#include "mainmenu.h"
#include "mainwindow.h"
#include "kernelwindow.h"

MainMenu::MainMenu(QWidget *parent) : QWidget(parent) {
    titleLabel = new QLabel("Image Seg App", this);
    QFont titleFont = titleLabel->font();
    titleFont.setPointSize(18);
    titleFont.setBold(true);
    titleLabel->setFont(titleFont);
    titleLabel->setAlignment(Qt::AlignCenter);

    colorSegmentationButton = new QPushButton("Color Based Segmentation", this);
    kernelSegmentationButton = new QPushButton("Kernel Based Segmentation", this);

    colorSegmentationButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    kernelSegmentationButton->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    colorSegmentationButton->setFixedHeight(50);
    kernelSegmentationButton->setFixedHeight(50);

    // Create the layout
    QVBoxLayout *vLayout = new QVBoxLayout(this);
    vLayout->addWidget(titleLabel);
    vLayout->addSpacing(20);
    vLayout->addWidget(colorSegmentationButton);
    vLayout->addWidget(kernelSegmentationButton);
    vLayout->setAlignment(Qt::AlignCenter);

    // Connect the buttons to their respective slots
    connect(colorSegmentationButton, &QPushButton::clicked, this, &MainMenu::onColorBasedSegmentation);
    connect(kernelSegmentationButton, &QPushButton::clicked, this, &MainMenu::onKernelBasedSegmentation);
}

void MainMenu::onColorBasedSegmentation() {
    MainWindow *colorWindow = new MainWindow();
    colorWindow->show();
    this->close();
}

void MainMenu::onKernelBasedSegmentation() {
    KernelWindow *kernelWindow = new KernelWindow();
    kernelWindow->show();
    this->close();
}
