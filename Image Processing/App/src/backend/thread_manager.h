#ifndef THREAD_MANAGER_H
#define THREAD_MANAGER_H

#include <QObject>
#include <QThread>

#include "file_capture.h"

class Thread_Manager : public QObject {
        Q_OBJECT
        QThread video_thread;
        File_Capture video_capture;
    public:
        explicit Thread_Manager(QObject *parent = nullptr);
        Q_INVOKABLE void run_capture(QString url);
    signals:
        void update_view(cv::Mat frame);
    
    public slots:
        void receive_frame(cv::Mat frame);
};

#endif