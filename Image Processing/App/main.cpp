#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

double getContrast(const cv::Mat& image) {
    // Obtenemos valores minimos y máximos de la matriz de la imagen para obtener su contraste
	double minVal, maxVal;
	cv::minMaxLoc(image, &minVal, &maxVal);
	return maxVal - minVal;
}

double laplacianVariance(const cv::Mat& image) {
    cv::Mat laplacianImage;
    // Operador laplaciano
    cv::Laplacian(image, laplacianImage, CV_64F); 

    // Calculamos la media y desviación estándar
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacianImage, mean, stddev); 

    return stddev[0] * stddev[0]; 
}

void compare_contrast() {

    std::vector<std::string> paths = { "Resources/Imagen1.jpg", "Resources/Imagen2.jpg", "Resources/Imagen3.jpg" };
    std::vector<cv::Mat> images;
    std::vector<double> contrasts;

    for (const auto& path : paths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "Error loading image: " << path << std::endl;
            return;
        }
        images.push_back(img);
        contrasts.push_back(getContrast(img));
    }

    double maxContrast = 0;
    int indexMaxContrast = -1;
    for (int i = 0; i < contrasts.size(); i++) {
        if (contrasts[i] > maxContrast) {
            maxContrast = contrasts[i];
            indexMaxContrast = i;
        }
    }

    std::cout << "La imagen con mejor contraste es: " << paths[indexMaxContrast] << " con un contraste de " << maxContrast << std::endl;
}

void compare_focus() {
    std::vector<std::string> paths = { "Resources/Imagen4.jpg", "Resources/Imagen5.jpg", "Resources/Imagen6.jpg" };
    std::vector<cv::Mat> images;
    std::vector<double> variances;

    for (const auto& path : paths) {
        cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cout << "Error loading image: " << path << std::endl;
            continue; 
        }
        images.push_back(img);
        variances.push_back(laplacianVariance(img));
    }

    double maxVariance = 0;
    int indexMaxVariance = -1;
    for (int i = 0; i < variances.size(); i++) {
        if (variances[i] > maxVariance) {
            maxVariance = variances[i];
            indexMaxVariance = i;
        }
    }

    if (indexMaxVariance != -1) {
        std::cout << "La imagen con mejor enfoque es: " << paths[indexMaxVariance]
            << " con una varianza de Laplaciano de " << maxVariance << std::endl;
    }
    else {
        std::cout << "No se pudo determinar una imagen con mejor enfoque." << std::endl;
    }
}
int main() {
    compare_contrast();
    compare_focus();

	return 0;
}