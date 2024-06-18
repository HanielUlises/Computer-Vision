#include <QApplication>
#include "mainmenu.h"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);

    MainMenu menu;
    menu.setWindowTitle("Image Segmentation App");
    menu.resize(800, 600);
    menu.show();

    return app.exec();
}
