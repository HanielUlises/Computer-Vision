#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMenu>
#include <QToolBar>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QStatusBar>
#include <QLabel>
#include <QAction>  

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void openImage();

private:
    void initUI();

private:
    QMenu *fileMenu;
    QMenu *viewMenu;
    QToolBar *fileToolBar;
    QToolBar *viewToolBar;
    QGraphicsScene *imageScene;
    QGraphicsView *imageView;
    QStatusBar *mainStatusBar;
    QLabel *mainStatusLabel;
    QAction *openAction;  
};

#endif // MAINWINDOW_H
