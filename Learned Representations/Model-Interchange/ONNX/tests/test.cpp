#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "onnx_inference_engine.hpp"
#include "preprocessor.hpp"

#include <opencv2/imgproc.hpp>

using namespace hpcv;

TEST_CASE("Letterbox preserves aspect ratio", "[preprocessor]")
{
    cv::Mat src(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    letterbox_info info;
    cv::Mat out = preprocessor::letterbox(src, {640, 640}, &info);

    REQUIRE(out.size() == cv::Size(640, 640));
    REQUIRE(info.scale == Approx(1.0f));
    REQUIRE(info.pad_left == 0);
    REQUIRE(info.pad_top  == 80); 
}

TEST_CASE("toBlob produces correct shape and range", "[preprocessor]")
{
    cv::Mat rgb(224, 224, CV_8UC3, cv::Scalar(255, 128, 0));
    auto blob = preprocessor::to_blob(rgb, norm_zero_one());

    REQUIRE((int)blob.size() == 3 * 224 * 224);

    REQUIRE(blob[0] == Approx(1.0f).margin(1e-4f));

    REQUIRE(blob[224*224] == Approx(128.f / 255.f).margin(1e-4f));
}

TEST_CASE("toBatchBlob: N * C * H * W", "[preprocessor]")
{
    std::vector<cv::Mat> batch = {
        cv::Mat(64, 64, CV_8UC3, cv::Scalar(255, 0, 0)),
        cv::Mat(64, 64, CV_8UC3, cv::Scalar(0, 255, 0)),
    };
    auto blob = preprocessor::to_batch_blob(batch, norm_zero_one());
    REQUIRE((int)blob.size() == 2 * 3 * 64 * 64);
}

TEST_CASE("unletterbox round-trips bbox", "[preprocessor]")
{
    letterbox_info info;
    info.scale    = 0.75f;
    info.pad_left  = 0;
    info.pad_top   = 40;
    info.orig_size = {640, 480};

    cv::Rect2f box_lb{10.f, 50.f, 100.f, 80.f};
    auto result = preprocessor::unletterbox(box_lb, info);

    float exp_x = (10.f - 0.f)  / 0.75f / 640.f;
    REQUIRE(result.x == Approx(exp_x).margin(1e-4f));
}

// Smoke test: engine construction (CPU, dummy model)
// Requires a valid ONNX model at HPCV_TEST_MODEL env variable btw
TEST_CASE("Engine constructs on CPU with a real model", "[engine][integration]")
{
    const char* model_path = std::getenv("HPCV_TEST_MODEL");
    if (!model_path) {
        WARN("HPCV_TEST_MODEL not set — skipping engine integration test.");
        return;
    }

    onnx_inference_engine::config cfg;
    cfg.model_path = model_path;
    cfg.provider  = execution_provider::cpu;

    REQUIRE_NOTHROW([&]{ onnx_inference_engine e(cfg); }());
}

TEST_CASE("detect returns empty on blank image (CPU)", "[engine][integration]")
{
    const char* model_path = std::getenv("HPCV_TEST_MODEL");
    if (!model_path) {
        WARN("HPCV_TEST_MODEL not set — skipping.");
        return;
    }

    onnx_inference_engine::config cfg;
    cfg.model_path = model_path;
    cfg.provider  = execution_provider::cpu;
    onnx_inference_engine engine(cfg);

    // Solid grey image should NOT produce high-confidence detections </3
    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    auto dets = engine.detect(blank, 0.9f, 0.45f);

    REQUIRE(dets.size() < 5);
}