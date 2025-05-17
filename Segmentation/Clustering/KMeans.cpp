#include "KMeans.hpp"
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <limits>

template <typename T>
KMeans<T>::KMeans(int k, int max_iter, double tol)
    : k_(k), max_iter_(max_iter), tol_(tol) {}

template <typename T>
void KMeans<T>::fit(const std::vector<cv::Mat>& descriptors) {
    std::vector<Eigen::VectorXd> dataPoints;
    for (const auto& desc : descriptors) {
        for (int i = 0; i < desc.rows; ++i) {
            Eigen::VectorXd point(desc.cols);
            for (int j = 0; j < desc.cols; ++j) {
                point(j) = desc.at<float>(i, j);
            }
            dataPoints.push_back(point);
        }
    }

    // Initialize centroids randomly from the data points
    centroids_ = Eigen::MatrixXd(k_, dataPoints[0].size());
    for (int i = 0; i < k_; ++i) {
        centroids_.row(i) = dataPoints[rand() % dataPoints.size()].transpose();
    }

    std::vector<int> labels(dataPoints.size(), -1);
    bool converged = false;
    int iter = 0;

    while (!converged && iter < max_iter_) {
        std::vector<int> newLabels(dataPoints.size());
        converged = true;

        // Step 1: Assign each point to the nearest centroid
        for (size_t i = 0; i < dataPoints.size(); ++i) {
            int closestCentroid = -1;
            double minDist = std::numeric_limits<double>::max();
            for (int j = 0; j < k_; ++j) {
                double dist = compute_distance(dataPoints[i], centroids_.row(j).transpose());
                if (dist < minDist) {
                    minDist = dist;
                    closestCentroid = j;
                }
            }
            newLabels[i] = closestCentroid;
            if (newLabels[i] != labels[i]) {
                converged = false;
            }
        }

        // Step 2: Update centroids
        update_centroids(descriptors, newLabels);
        labels = newLabels;
        ++iter;
    }
}

template <typename T>
std::vector<int> KMeans<T>::predict(const std::vector<cv::Mat>& descriptors) {
    std::vector<int> predictions;
    std::vector<Eigen::VectorXd> dataPoints;
    
    // Convert descriptors to Eigen matrix form
    for (const auto& desc : descriptors) {
        for (int i = 0; i < desc.rows; ++i) {
            Eigen::VectorXd point(desc.cols);
            for (int j = 0; j < desc.cols; ++j) {
                point(j) = desc.at<float>(i, j);
            }
            dataPoints.push_back(point);
        }
    }

    // Assign points to nearest centroid
    for (const auto& point : dataPoints) {
        int closestCentroid = -1;
        double minDist = std::numeric_limits<double>::max();
        for (int i = 0; i < k_; ++i) {
            double dist = compute_distance(point, centroids_.row(i).transpose());
            if (dist < minDist) {
                minDist = dist;
                closestCentroid = i;
            }
        }
        predictions.push_back(closestCentroid);
    }

    return predictions;
}

template <typename T>
Eigen::MatrixXd KMeans<T>::get_centroids() const {
    return centroids_;
}

template <typename T>
double KMeans<T>::compute_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b) {
    return (a - b).norm();
}

template <typename T>
void KMeans<T>::update_centroids(const std::vector<cv::Mat>& descriptors, const std::vector<int>& labels) {
    std::vector<Eigen::VectorXd> newCentroids(k_, Eigen::VectorXd(descriptors[0].cols));
    std::vector<int> counts(k_, 0);

    // Accumulate points for each centroid
    for (size_t i = 0; i < descriptors.size(); ++i) {
        Eigen::VectorXd point(descriptors[i].cols);
        for (int j = 0; j < descriptors[i].cols; ++j) {
            point(j) = descriptors[i].at<float>(i, j);
        }
        newCentroids[labels[i]] += point;
        counts[labels[i]]++;
    }

    // Compute the mean of points assigned to each centroid
    for (int i = 0; i < k_; ++i) {
        if (counts[i] > 0) {
            newCentroids[i] /= counts[i];
        }
        centroids_.row(i) = newCentroids[i].transpose();
    }
}
