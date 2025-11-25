#include "thread_manager.h"

Thread_Manager::Thread_Manager(QObject *parent) : QObject(parent) {

}

void Thread_Manager::run_capture(QString url) {
    video_capture.set_url(url);

    connect(&video_thread, &QThread::started, &video_capture, &File_Capture::open_video);
    connect(&video_thread, &QThread::finished, &video_capture, &File_Capture::deleteLater);
    connect(&video_capture, &File_Capture::new_frame_captured, this, &Thread_Manager::receive_frame);

    video_capture.moveToThread(&video_thread);
    video_thread.start();
}

void Thread_Manager::receive_frame(cv::Mat frame){
    emit(update_view(frame));
}