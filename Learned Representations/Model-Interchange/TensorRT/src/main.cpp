#include "trt_inference_engine.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <iostream>
#include <string>

struct args {
    std::string engine;         // .engine file  (preferred)
    std::string onnx;           // .onnx file     (builds engine on first run)
    std::string save_engine;    // where to serialise a freshly-built engine
    std::string labels;
    std::string source;
    float conf_thresh = 0.5f;
    float nms_thresh  = 0.45f;
    bool  fp16        = true;
    int   device      = 0;
    int   max_batch   = 1;
};

static args parse_args(int argc, char** argv)
{
    args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if      (k == "--engine"      && i+1 < argc) a.engine       = argv[++i];
        else if (k == "--onnx"        && i+1 < argc) a.onnx         = argv[++i];
        else if (k == "--save-engine" && i+1 < argc) a.save_engine  = argv[++i];
        else if (k == "--labels"      && i+1 < argc) a.labels       = argv[++i];
        else if (k == "--source"      && i+1 < argc) a.source       = argv[++i];
        else if (k == "--conf"        && i+1 < argc) a.conf_thresh  = std::stof(argv[++i]);
        else if (k == "--nms"         && i+1 < argc) a.nms_thresh   = std::stof(argv[++i]);
        else if (k == "--device"      && i+1 < argc) a.device       = std::stoi(argv[++i]);
        else if (k == "--max-batch"   && i+1 < argc) a.max_batch    = std::stoi(argv[++i]);
        else if (k == "--no-fp16")                   a.fp16         = false;
    }
    if (a.engine.empty() && a.onnx.empty()) {
        std::cerr <<
            "Usage: trt_demo --engine model.engine  [OR --onnx model.onnx]\n"
            "                --source <image|video|0>\n"
            "                [--labels labels.txt]\n"
            "                [--save-engine out.engine]   # save newly-built engine\n"
            "                [--conf 0.5] [--nms 0.45]\n"
            "                [--no-fp16] [--device 0] [--max-batch 1]\n";
        std::exit(1);
    }
    return a;
}

static void draw(cv::Mat& frame, const std::vector<hpcv::detection>& dets)
{
    for (auto& d : dets) {
        int x  = (int)(d.bbox.x * frame.cols);
        int y  = (int)(d.bbox.y * frame.rows);
        int bw = (int)(d.bbox.width  * frame.cols);
        int bh = (int)(d.bbox.height * frame.rows);

        cv::Scalar col(0, 200, 50);
        cv::rectangle(frame, {x, y, bw, bh}, col, 2);

        std::string lbl = d.label + " " +
                          std::to_string((int)(d.confidence * 100)) + "%";
        int base;
        auto ts = cv::getTextSize(lbl, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &base);
        cv::rectangle(frame, {x, y - ts.height - 6, ts.width + 4, ts.height + 6},
                      col, cv::FILLED);
        cv::putText(frame, lbl, {x + 2, y - 4},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, {0,0,0}, 1, cv::LINE_AA);
    }
}

int main(int argc, char** argv)
{
    args args = parse_args(argc, argv);

    hpcv::trt_inference_engine::config cfg;
    cfg.engine_path    = args.engine;
    cfg.onnx_path      = args.onnx;
    cfg.fp16           = args.fp16;
    cfg.device_id      = args.device;
    cfg.max_batch_size = args.max_batch;
    if (!args.save_engine.empty())
        cfg.engine_path = args.save_engine;

    hpcv::trt_inference_engine engine(cfg);
    if (!args.labels.empty())
        engine.load_labels(args.labels);

    bool is_video = false;
    cv::VideoCapture cap;
    cv::Mat image;

    try {
        int cam = std::stoi(args.source);
        cap.open(cam);
        is_video = true;
    } catch (...) {
        image = cv::imread(args.source);
        if (image.empty()) {
            cap.open(args.source);
            is_video = cap.isOpened();
            if (!is_video) {
                std::cerr << "[hpcv/trt] Cannot open: " << args.source << "\n";
                return 1;
            }
        }
    }

    auto run_frame = [&](cv::Mat& frame) {
        auto t0  = std::chrono::steady_clock::now();
        auto det = engine.detect(frame, args.conf_thresh, args.nms_thresh);
        auto t1  = std::chrono::steady_clock::now();

        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        draw(frame, det);
        std::string info = "TRT  " + std::to_string((int)ms) + " ms";
        cv::putText(frame, info, {10, 28},
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 255}, 2, cv::LINE_AA);
        cv::imshow("hpcv TensorRT", frame);
    };

    if (is_video) {
        cv::Mat frame;
        while (cap.read(frame)) {
            run_frame(frame);
            if (cv::waitKey(1) == 'q') break;
        }
    } else {
        run_frame(image);
        cv::waitKey(0);
    }
    return 0;
}
