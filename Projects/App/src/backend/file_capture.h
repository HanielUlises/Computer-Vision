#ifndef FILE_CAPTURE_H
#define FILE_CAPTURE_H

#include <vector>
#include <string>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/dnn.hpp>

#include <QObject>
#include <QUrl>
#include <QDebug>
#include <QThread>

class File_Capture : public QObject{
    Q_OBJECT
    cv::VideoCapture *video_capture;
    QString url;
    bool stop_video;
    
    public:
        explicit File_Capture(QObject *parent = nullptr);
        void set_url(QString &value);
        void find_objects(cv::Mat &frame, std::vector<cv::Mat> &outputs, cv::dnn::Net &net, std::vector<std::string> &classes);
        void draw_obj_box(int class_id, int left, int top, int right, int bottom, cv::Mat &frame, std::vector<std::string> &classes);

    signals:
        void new_frame_captured(cv::Mat frame);
    
    public slots:
        bool open_video();
        bool subtract_background();
        void detect_color();
        void open_camera();
        void detect_faces();
        void track_yolo();
};

#endif