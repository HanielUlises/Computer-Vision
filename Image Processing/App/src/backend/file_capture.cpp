#include "file_capture.h"


File_Capture::File_Capture(QObject *parent) : QObject(parent) {
    stop_video = false;
}

void File_Capture::set_url (QString &value){
    value.remove("file://");
    url = value;
}

bool File_Capture::open_video(){
    cv::Mat raw_frame;
    video_capture = new cv::VideoCapture;
    video_capture -> open(url.toStdString());

    if(!video_capture -> isOpened()) return false;

    while(video_capture -> read(raw_frame) && !stop_video) {
        emit(new_frame_captured(raw_frame));
    }
}


