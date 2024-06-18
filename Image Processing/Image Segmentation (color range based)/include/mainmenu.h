#ifndef MAINMENU_H
#define MAINMENU_H

#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QLabel>

class MainMenu : public QWidget {
    Q_OBJECT

public:
    MainMenu(QWidget *parent = nullptr);
    ~MainMenu() = default;

private slots:
    void onColorBasedSegmentation();
    void onKernelBasedSegmentation();

private:
    QLabel *titleLabel;
    QPushButton *colorSegmentationButton;
    QPushButton *kernelSegmentationButton;
};

#endif // MAINMENU_H
