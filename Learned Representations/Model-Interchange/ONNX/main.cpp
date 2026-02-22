#include "onnx_inference_engine.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <iostream>
#include <string>

struct args {
    std::string model;
    std::string labels;
    std::string source; 
    float       conf_thresh = 0.5f;
    float       nms_thresh  = 0.45f;
    bool        cuda       = true;
    int         device     = 0;
};

static args parse_args(int argc, char** argv)
{
    args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--model"  && i+1 < argc) a.model  = argv[++i];
        else if (k == "--labels" && i+1 < argc) a.labels = argv[++i];
        else if (k == "--source" && i+1 < argc) a.source = argv[++i];
        else if (k == "--conf"   && i+1 < argc) a.conf_thresh = std::stof(argv[++i]);
        else if (k == "--nms"    && i+1 < argc) a.nms_thresh  = std::stof(argv[++i]);
        else if (k == "--cpu")                  a.cuda = false;
        else if (k == "--device" && i+1 < argc) a.device = std::stoi(argv[++i]);
    }
    if (a.model.empty()) {
        std::cerr << "Usage: onnx_demo --model <path.onnx> --source <img|0> "
                     "[--labels labels.txt] [--conf 0.5] [--nms 0.45] "
                     "[--cpu] [--device 0]\n";
        std::exit(1);
    }
    return a;
}

static void draw_detections(cv::Mat& frame,
                            const std::vector<hpcv::detection>& dets)
{
    for (auto& d : dets) {
        int x  = (int)(d.bbox.x * frame.cols);
        int y  = (int)(d.bbox.y * frame.rows);
        int bw = (int)(d.bbox.width  * frame.cols);
        int bh = (int)(d.bbox.height * frame.rows);

        cv::Scalar colour(0, 255, 0);
        cv::rectangle(frame, {x, y, bw, bh}, colour, 2);

        std::string label = d.label + " " +
                            std::to_string((int)(d.confidence * 100)) + "%";
        int   baseline;
        auto  text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                      0.55, 1, &baseline);
        cv::rectangle(frame, {x, y - text_size.height - 6, text_size.width + 4, text_size.height + 6},
                      colour, cv::FILLED);
        cv::putText(frame, label, {x + 2, y - 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, {0, 0, 0}, 1, cv::LINE_AA);
    }
}

int main(int argc, char** argv)
{
    args args = parse_args(argc, argv);

    hpcv::onnx_inference_engine::config cfg;
    cfg.model_path  = args.model;
    cfg.provider   = args.cuda ? hpcv::execution_provider::cuda
                               : hpcv::execution_provider::cpu;
    cfg.device_id   = args.device;

    hpcv::onnx_inference_engine engine(cfg);

    if (!args.labels.empty())
        engine.load_labels(args.labels);

    bool is_video = false;
    cv::VideoCapture cap;
    cv::Mat image;

    try {
        int cam_id = std::stoi(args.source);
        cap.open(cam_id);
        is_video = true;
    } catch (...) {
        image = cv::imread(args.source);
        if (image.empty()) {
            cap.open(args.source);
            is_video = cap.isOpened();
            if (!is_video && image.empty()) {
                std::cerr << "[hpcv] Cannot open source: " << args.source << "\n";
                return 1;
            }
        }
    }

    auto run_on_frame = [&](cv::Mat& frame) {
        auto t0 = std::chrono::steady_clock::now();
        auto dets = engine.detect(frame, args.conf_thresh, args.nms_thresh);
        auto t1 = std::chrono::steady_clock::now();

        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        draw_detections(frame, dets);

        std::string fps = "Inference: " + std::to_string((int)ms) + " ms";
        cv::putText(frame, fps, {10, 28},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2, cv::LINE_AA);

        cv::imshow("hpcv ONNX", frame);
        return (int)dets.size();
    };

    if (is_video) {
        cv::Mat frame;
        while (cap.read(frame)) {
            run_on_frame(frame);
            if (cv::waitKey(1) == 'q') break;
        }
    } else {
        run_on_frame(image);
        cv::waitKey(0);
    }

    return 0;
}