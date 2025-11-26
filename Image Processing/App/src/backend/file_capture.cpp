#include "file_capture.h"


File_Capture::File_Capture(QObject *parent) : QObject(parent) {
    stop_video = false;
}

void File_Capture::set_url (QString &value){
    QUrl qurl(value);
    url = qurl.toLocalFile(); 
    qDebug() << "[File_Capture] final URL =" << url;
}

bool File_Capture::open_video(){
    cv::Mat raw_frame;
    video_capture = new cv::VideoCapture;
    video_capture -> open(url.toStdString());

    if(!video_capture -> isOpened()) return false;

    while(video_capture -> read(raw_frame) && !stop_video) {
        emit(new_frame_captured(raw_frame));
    }
    
    return true;
}

bool File_Capture::subtract_background() {
    cv::Ptr<cv::BackgroundSubtractor> subtraction_ptr;
    subtraction_ptr = cv::createBackgroundSubtractorMOG2();

    cv::Mat raw_frame;
    cv::Mat background_mask;

    video_capture = new cv::VideoCapture;
    video_capture -> open(url.toStdString());

    if(!video_capture -> isOpened()) {
        return false;
    }

    while(video_capture -> read(raw_frame)) {
        subtraction_ptr -> apply(raw_frame, background_mask);
        emit(new_frame_captured(background_mask));
    }
    return true;
}
