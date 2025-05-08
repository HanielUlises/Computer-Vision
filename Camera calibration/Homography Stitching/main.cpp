#include "homography_stitcher.hpp"
#include <iostream>
#include <format>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " <image_1> <image_2> <output_image>\n";
        return 1;
    }
    try {
        homography_stitching::homography_stitcher stitcher(argv[1], argv[2]);
        auto result_opt = stitcher.stitch_images();
        if (!result_opt) {
            std::cerr << "error: " << result_opt.error() << "\n";
            return 1;
        }
        auto& result = *result_opt;
        std::cout << std::format("inlier_count: {}\n", result.inlier_count);
        std::cout << "homography_matrix:\n";
        for (int i = 0; i < 3; ++i) {
            std::cout << std::format("[{:.4f}, {:.4f}, {:.4f}]\n",
                result.homography_matrix.at<double>(i, 0),
                result.homography_matrix.at<double>(i, 1),
                result.homography_matrix.at<double>(i, 2));
        }
        stitcher.save_stitched_image(argv[3]);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}