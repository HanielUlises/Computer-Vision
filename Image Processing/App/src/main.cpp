#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>

#include "backend/frame.h"
#include "backend/thread_manager.h"

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);
    QGuiApplication app(argc, argv);
    qRegisterMetaType<Frame>("Frame");
    qRegisterMetaType<cv::Mat>("Mat");
    qmlRegisterType<Frame>("com.app.frame", 1, 0, "Frame");

    Thread_Manager* video_thread = new Thread_Manager(&app);

    QQmlApplicationEngine engine;

    engine.rootContext()->setContextProperty("video_thread", video_thread);
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}