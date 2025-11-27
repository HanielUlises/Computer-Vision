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

void File_Capture::detect_color() {
    cv::Mat raw_frame;
    cv::Mat hsv_frame;
    cv::Mat threshold_frame;

    video_capture = new cv::VideoCapture;
    video_capture -> open(url.toStdString());

    if(!video_capture -> isOpened()) {
        return;
    }

    while(video_capture -> read(raw_frame)) {
        cv::cvtColor(raw_frame, hsv_frame, cv::COLOR_BGR2HSV);
        cv::inRange(hsv_frame, cv::Scalar(218, 67, 100), cv::Scalar(240, 96, 100), threshold_frame);
        emit(new_frame_captured(threshold_frame));
    }

}

void File_Capture::open_camera() {
    cv::Mat raw_frame;

    video_capture = new cv::VideoCapture;
    video_capture -> open(0);

    if(!video_capture -> isOpened())
        return;

    while(video_capture -> read(raw_frame) && !stop_video)
        emit(new_frame_captured(raw_frame));
}

void File_Capture::detect_faces() {
    cv::CascadeClassifier face_classifier;

    if(!face_classifier.load("../../build/haarcascade_frontalface_alt_tree.xml")) {
        return;
    }

    cv::Mat raw_frame;
    cv::Mat gray_mask;

    video_capture = new cv::VideoCapture;
    video_capture -> open(url.toStdString());

    if(!video_capture -> isOpened()) return;

    while(video_capture -> read(raw_frame)) {
        cv::cvtColor(raw_frame, gray_mask, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray_mask, gray_mask);

        std::vector<cv::Rect> faces;
        face_classifier.detectMultiScale(gray_mask, faces);

        for(unsigned int i = 0; i < faces.size(); i++) {
            cv::Point center(faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2);
            cv::ellipse(raw_frame, center, cv::Size(faces[i].width/2, faces[i].height/2), 
                        0, 0, 360, cv::Scalar(255, 255, 0, 4));
        }

        emit(new_frame_captured(raw_frame));
    }
}