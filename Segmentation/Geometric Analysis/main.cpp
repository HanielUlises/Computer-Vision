#include "geometric_shape_detector.hpp"
#include <iostream>
#include <format>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " <input_image> <output_image>\n";
        return 1;
    }
    try {
        geometric_analysis::geometric_shape_detector detector(argv[1]);
        auto shapes = detector.detect_shapes();
        for (const auto& [index, shape] : std::views::enumerate(shapes)) {
            std::cout << std::format(
                "shape_{}: type={}, area={:.2f}, perimeter={:.2f}, centroid=({:.2f}, {:.2f})\n",
                index, shape.type, shape.area, shape.perimeter, shape.centroid.x, shape.centroid.y
            );
        }
        detector.save_annotated_image(argv[2]);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}