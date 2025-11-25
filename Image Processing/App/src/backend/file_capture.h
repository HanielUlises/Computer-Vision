#ifndef FILE_CAPTURE_H
#define FILE_CAPTURE_H

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <QObject>

class File_Capture : public QObject{
    Q_OBJECT
    cv::VideoCapture *video_capture;
    QString url;
    bool stop_video;
    
    public:
        explicit File_Capture(QObject *parent = nullptr);
        void set_url(QString &value);

    signals:
        void new_frame_captured(cv::Mat frame);
    
    public slots:
        bool open_video();
};

#endif