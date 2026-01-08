#include "pyramidal_lk.hpp"

PyramidalLK::PyramidalLK(
    int win_size,
    int max_level,
    int max_iters,
    double epsilon
)
    : win_size_(win_size, win_size),
      max_level_(max_level),
      termcrit_(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                max_iters, epsilon) {}

void PyramidalLK::track(
    const cv::Mat& prev_img,
    const cv::Mat& curr_img,
    std::vector<cv::Point2f>& prev_pts,
    std::vector<cv::Point2f>& curr_pts,
    std::vector<uchar>& status
) {
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(
        prev_img,
        curr_img,
        prev_pts,
        curr_pts,
        status,
        err,
        win_size_,
        max_level_,
        termcrit_,
        cv::OPTFLOW_USE_INITIAL_FLOW
    );
}
