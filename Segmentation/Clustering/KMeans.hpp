#ifndef KMEANS_H
#define KMEANS_H

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>

class KMeans {
public:
    KMeans(int k, int max_iter = 100, double tol = 1e-4);

    void fit(const std::vector<cv::Mat>& descriptors);
    std::vector<int> predict(const std::vector<cv::Mat>& descriptors);
    Eigen::MatrixXd get_centroids() const;

private:
    int k_;                    // Number of clusters (k)
    int max_iter_;             // Maximum number of iterations
    double tol_;               // Convergence tolerance (not used explicitly here)
    Eigen::MatrixXd centroids_; // Matrix of shape (k, d) containing centroids

    double compute_distance(const Eigen::VectorXd& a, const Eigen::VectorXd& b);
    void update_centroids(const std::vector<Eigen::VectorXd>& points, const std::vector<int>& labels);
    std::vector<Eigen::VectorXd> convert_to_vectors(const std::vector<cv::Mat>& descriptors);
};

#endif // KMEANS_H
