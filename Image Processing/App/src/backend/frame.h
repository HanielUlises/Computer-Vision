#ifndef FRAME_H
#define FRAME_H

#include <QObject>  
#include <QQuickPaintedItem>
#include <QImage>
#include <QPainter>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <vector>

class Frame : public QQuickPaintedItem{
        Q_OBJECT
        Q_PROPERTY(QImage frame READ get_frame WRITE set_frame NOTIFY frame_changed)

        cv::Mat raw_frame;
        QImage frame;

    public:
        Frame(QQuickPaintedItem *parent = 0);
        
        void paint(QPainter *painter);
        Q_INVOKABLE void set_raw_frame(const cv::Mat &value);
        void set_frame(const QImage &value);
        cv::Mat get_raw_frame() const;
        QImage get_frame() const;

        Q_INVOKABLE void open_image(QString url);   

        // Gaussian blur
        Q_INVOKABLE bool smooth_image();
        // Morphological operations
        Q_INVOKABLE bool erode_image();
        Q_INVOKABLE bool dilate_image();
        // Edge detectors
        Q_INVOKABLE bool find_cotours();
        Q_INVOKABLE bool find_horizontal_lines();
        Q_INVOKABLE bool find_vertical_lines();
    signals:
        void frame_changed();
    
    public slots:
        void convert_to_QImage();
        
};

#endif