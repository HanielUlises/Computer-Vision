#include "frame.h"

Frame::Frame(QQuickPaintedItem *parent) : QQuickPaintedItem(parent) {

}

void Frame::paint(QPainter *painter) {
    frame.scaled(640, 480, Qt::IgnoreAspectRatio, Qt::FastTransformation);
    painter -> drawImage(0, 0, frame, 0, 0, -1, -1, Qt::AutoColor);
}

void Frame::set_raw_frame(const cv::Mat &value) {
    raw_frame = value;
    convert_to_QImage();
}

void Frame::set_frame(const QImage &value) {
    frame = value;
    Frame::update();
    emit(frame_changed());
}

cv::Mat Frame::get_raw_frame() const{
    return raw_frame;
}

QImage Frame::get_frame() const{
    return frame;
}

void Frame::open_image(QString url) {
    url.remove("file://");
    raw_frame = cv::imread(url.toStdString());
    convert_to_QImage();
}

void Frame::convert_to_QImage() {
    cv::Mat temp_mat;
    cv::cvtColor(raw_frame, temp_mat, cv::COLOR_BGR2RGB);
    QImage temp_image(static_cast<uchar*>(temp_mat.data), temp_mat.cols, temp_mat.rows, temp_mat.step, QImage::Format_RGB888);

    frame = temp_image;
    frame.detach();
    Frame::update();
    emit(frame_changed());
}