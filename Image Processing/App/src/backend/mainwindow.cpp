#include "mainwindow.h"

#include <QMenuBar>
#include <QAction>
#include <QDebug>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    initUI();
}

MainWindow::~MainWindow() {
}

void MainWindow::initUI() {
    this->resize(800, 600);

    fileMenu = menuBar()->addMenu("&File");
    viewMenu = menuBar()->addMenu("&View");

    fileToolBar = addToolBar("File");
    viewToolBar = addToolBar("View");

    openAction = new QAction("Open", this);
    fileMenu->addAction(openAction);
    fileToolBar->addAction(openAction);

    imageScene = new QGraphicsScene(this);
    imageView = new QGraphicsView(imageScene);
    setCentralWidget(imageView);

    mainStatusBar = statusBar();
    mainStatusLabel = new QLabel(mainStatusBar);
    mainStatusBar->addPermanentWidget(mainStatusLabel);
    mainStatusLabel->setText("Image Information will be here!");

    connect(openAction, &QAction::triggered, this, &MainWindow::openImage);
}

void MainWindow::openImage() {
    qDebug() << "slot openImage is called.";
}
