#include "image_processor.h"

QImage ImageProcessor::extractRedChannel(const QImage &image) {
    QImage redImage = image;
    for (int y = 0; y < redImage.height(); y++) {
        for (int x = 0; x < redImage.width(); x++) {
            int red = QColor(redImage.pixel(x, y)).red();
            redImage.setPixel(x, y, qRgb(red, 0, 0));
        }
    }
    return redImage;
}

QImage ImageProcessor::extractGreenChannel(const QImage &image) {
    QImage greenImage = image;
    for (int y = 0; y < greenImage.height(); y++) {
        for (int x = 0; x < greenImage.width(); x++) {
            int green = QColor(greenImage.pixel(x, y)).green();
            greenImage.setPixel(x, y, qRgb(0, green, 0));
        }
    }
    return greenImage;
}

QImage ImageProcessor::extractBlueChannel(const QImage &image) {
    QImage blueImage = image;
    for (int y = 0; y < blueImage.height(); y++) {
        for (int x = 0; x < blueImage.width(); x++) {
            int blue = QColor(blueImage.pixel(x, y)).blue();
            blueImage.setPixel(x, y, qRgb(0, 0, blue));
        }
    }
    return blueImage;
}
