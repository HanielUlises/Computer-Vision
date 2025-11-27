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

void File_Capture::track_yolo(){
    cv::Mat raw_frame;
    cv::Mat blob;

    float scale = 0.00392;
    std::vector<std::string> classes;

    std::string model_config = "../../build/samples/data/dnn/yolov2.cfg";
    std::string model_weights = "../../build/samples/data/dnn/yolov2.weights";
    std::string classes_file_url = "../../build/samples/data/dnn/object_detection_classes_pascal_voc.txt";

    std::ifstream classes_file (classes_file_url);

    if(!classes_file.is_open()) return;

    std::string line;
    while(std::getline(classes_file, line)) {
        classes.push_back(line);
    }

    cv::dnn::Net net = cv::dnn::readNet(model_weights, model_config);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    std::vector<std::string> objects_name = net.getUnconnectedOutLayersNames();

    video_capture = new cv::VideoCapture;
    video_capture -> open(url.toStdString());

    if(!video_capture -> isOpened()) return;

    while(video_capture -> read(raw_frame)) {
        cv::resize(raw_frame, raw_frame, cv::Size(416, 416), 0, 0,  cv::INTER_CUBIC);
        cv::dnn::blobFromImage(raw_frame, blob, scale, cv::Size(raw_frame.cols, raw_frame.rows), cv::Scalar(0, 0, 0));
        
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, objects_name);

        find_objects(raw_frame, outputs, net, classes);
        emit(new_frame_captured(raw_frame));
    }
}

void File_Capture::find_objects(cv::Mat &frame, std::vector<cv::Mat> &outputs, cv::dnn::Net &net, std::vector<std::string> &classes) {
    std::vector<int> output_layers = net.getUnconnectedOutLayers();
    std::vector<int> class_ids;
    std::vector<float> credences;
    std::vector<cv::Rect> boxes;

    float conf_threshold = 0.5f;
    float nms_threshold = 0.4f;

    for(unsigned int i = 0; i < outputs.size(); i++) {
        float *data = reinterpret_cast<float*>(outputs[i].data);
        for(unsigned int j = 0; j < outputs[i].rows; j++, data += outputs[i].cols) {
            cv::Mat scores = outputs[i].row(j).colRange(5, outputs[i].cols);
            cv::Point class_id_point;

            double credence;

            cv::minMaxLoc(scores, 0, &credence, 0, &class_id_point);

            if(credence > conf_threshold) {
                int center_x = static_cast<int>(data[0] * frame.cols);
                int center_y = static_cast<int>(data[1] * frame.rows);

                int width = static_cast<int> (data[2] * frame.cols);
                int height = static_cast<int> (data[3] * frame.rows);

                int left = center_x - width / 2;
                int top = center_y - height / 2;

                class_ids.push_back(class_id_point.x);
                credences.push_back(static_cast<float> (credence));
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }

    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, credences, conf_threshold, nms_threshold, indices);

    for(size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        cv::Rect box = boxes[idx];

        draw_obj_box(class_ids[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame, classes);
    }

}

void File_Capture::draw_obj_box(int class_id, int left, int top, int right, int bottom, cv::Mat &frame, std::vector<std::string> &classes) {
    cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 0, 0));
    std::string label = classes[class_id];

    cv::putText(frame, label, cv::Point(left, top), cv::FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(255, 0, 0));
}