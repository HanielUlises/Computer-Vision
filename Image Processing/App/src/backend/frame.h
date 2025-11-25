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
    signals:
        void frame_changed();
    
    public slots:
        void convert_to_QImage();
        
};

#endif