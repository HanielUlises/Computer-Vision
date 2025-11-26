#include "frame.h"

Frame::Frame(QQuickPaintedItem *parent) : QQuickPaintedItem(parent) {

}

void Frame::paint(QPainter *painter) {

    if (frame.isNull()) {
        return;
    }

    QImage scaled_frame = frame.scaled(
        width(), 
        height(),
        Qt::IgnoreAspectRatio,
        Qt::FastTransformation
    );

    painter->drawImage(0, 0, scaled_frame);
}

void Frame::set_raw_frame(const cv::Mat &value) {

    raw_frame = value;
    convert_to_QImage();
}

void Frame::set_frame(const QImage &value) {

    frame = value;
    frame.detach();
    update();
    emit(frame_changed());
}

cv::Mat Frame::get_raw_frame() const {

    return raw_frame;
}

QImage Frame::get_frame() const {

    return frame;
}

void Frame::open_image(QString url) {

    url.remove("file://");
    raw_frame = cv::imread(url.toStdString());

    if (!raw_frame.empty()) {
        convert_to_QImage();
    }
}

void Frame::convert_to_QImage() {

    if (raw_frame.empty()) {
        return;
    }

    cv::Mat temp_mat;
    cv::cvtColor(raw_frame, temp_mat, cv::COLOR_BGR2RGB);

    QImage temp_image(
        static_cast<uchar*> (temp_mat.data),
        temp_mat.cols,
        temp_mat.rows,
        temp_mat.step,
        QImage::Format_RGB888
    );

    frame = temp_image.copy();

    update();
    emit(frame_changed());
}

bool Frame::smooth_image() {
    if(!raw_frame.empty()){
        cv::Mat temp;
        cv::GaussianBlur(raw_frame, temp, cv::Size(31, 31), 0, 0);
        set_raw_frame(temp);
        
        return true;
    }
    
    return false;
}

bool Frame::erode_image() {
    if(!raw_frame.empty()){
        cv::Mat temp;
        temp = raw_frame;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6,6));
        cv::erode(temp, temp, kernel);

        set_raw_frame(temp);
        
        return true;
    }
    
    return false;
}

bool Frame::dilate_image() {
    if(!raw_frame.empty()){
        cv::Mat temp;
        temp = raw_frame;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6,6));
        cv::dilate(temp, temp, kernel);

        set_raw_frame(temp);
        
        return true;
    }
    
    return false;
}

bool Frame::find_cotours(){
    if(!raw_frame.empty()){
        cv::Mat temp;
        cv::cvtColor(raw_frame, temp, cv::COLOR_RGB2GRAY);
        cv::Canny(temp, temp, 0, 150, 3);

        std::vector<std::vector<cv::Point>> contours; 
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(temp, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

        cv::Mat drawing = cv::Mat::zeros(temp.size(), CV_8UC3);

        for(size_t i = 0; i < contours.size(); i++) {
            cv::Scalar color = cv::Scalar(256, 26, 150);
            cv::drawContours(drawing, contours, static_cast<int>(i), color, 2, cv::LINE_8, hierarchy, 0);
        }

        set_raw_frame(drawing);

        return true;
    }
    
    return false;
}

bool Frame::find_horizontal_lines(){
    if(!raw_frame.empty()){
        cv::Mat temp;
        cv::Mat binary;

        cv::cvtColor(raw_frame, temp, cv::COLOR_BGR2GRAY);
        cv::adaptiveThreshold(temp, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -2);

        int structure_size = binary.cols/30;

        cv::Mat horizontal_structure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(structure_size, 1));
        cv::erode(binary, binary, horizontal_structure, cv::Point(-1, -1));
        cv::dilate(binary, binary, horizontal_structure, cv::Point(-1, -1));

        set_raw_frame(binary);
        return true;
    }
    
    return false;
} 

bool Frame::find_vertical_lines(){
    if(!raw_frame.empty()){
        cv::Mat temp;
        cv::Mat binary;

        cv::cvtColor(raw_frame, temp, cv::COLOR_BGR2GRAY);
        cv::adaptiveThreshold(temp, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -2);

        int structure_size = binary.rows/30;

        cv::Mat vertical_structure = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, structure_size));
        cv::erode(binary, binary, vertical_structure, cv::Point(-1, -1));
        cv::dilate(binary, binary, vertical_structure, cv::Point(-1, -1));

        set_raw_frame(binary);
        return true;
    }
    
    return false;
}

bool Frame::laplacian_filter() {
    if(!raw_frame.empty()){
        cv::Mat temp;
        cv::Mat dest;

        cv::cvtColor(raw_frame, temp, cv::COLOR_BGR2GRAY);
        cv::Laplacian(temp, dest, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT);
        cv::convertScaleAbs(dest, dest);

        set_raw_frame(dest);
        return true;
    }

    return false;
}

bool Frame::apply_canny() {
    if(!raw_frame.empty()){
        cv::Mat temp;

        cv::cvtColor(raw_frame, temp, cv::COLOR_BGR2GRAY);
        cv::Canny(temp, temp, 0, 150, 3);

        set_raw_frame(temp);
        return true;
    }

    return false;
}