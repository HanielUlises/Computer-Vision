#ifndef KMEANS_H
#define KMEANS_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>

template <typename T>
class KMeans {
public:
    KMeans(int k, int maxIter = 100, double tol = 1e-4);
    void fit(const std::vector<cv::Mat>& descriptors);
    std::vector<int> predict(const std::vector<cv::Mat>& descriptors);
    Eigen::MatrixXd get_centroids() const;

private:
    int k_;  
    int max_iter_; 
    double tol_; 
    Eigen::MatrixXd centroids_;  

    double compute_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b);
    void update_centroids(const std::vector<cv::Mat>& descriptors, const std::vector<int>& labels);
};

#endif // KMEANS_H
