#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include <QImage>

class ImageProcessor {
public:
    static QImage extractRedChannel(const QImage &image);
    static QImage extractGreenChannel(const QImage &image);
    static QImage extractBlueChannel(const QImage &image);
};

#endif // IMAGEPROCESSOR_H
