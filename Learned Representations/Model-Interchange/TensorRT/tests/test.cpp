#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "trt_inference_engine.hpp"
#include "preprocessor.hpp"

#include <opencv2/imgproc.hpp>

using namespace hpcv;

TEST_CASE("Letterbox preserves aspect ratio", "[preprocessor]")
{
    cv::Mat src(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    letterbox_info info;
    auto out = preprocessor::letterbox(src, {640, 640}, &info);

    REQUIRE(out.size() == cv::Size(640, 640));
    REQUIRE(info.scale    == Approx(1.0f));
    REQUIRE(info.pad_left == 0);
    REQUIRE(info.pad_top  == 80);
}

TEST_CASE("to_blob produces correct size and values", "[preprocessor]")
{
    cv::Mat rgb(224, 224, CV_8UC3, cv::Scalar(255, 128, 0));
    auto blob = preprocessor::to_blob(rgb, norm_zero_one());

    REQUIRE((int)blob.size() == 3 * 224 * 224);
    REQUIRE(blob[0] == Approx(1.0f).margin(1e-4f));
    REQUIRE(blob[224*224] == Approx(128.f / 255.f).margin(1e-4f));
}

TEST_CASE("to_batch_blob: size == N*C*H*W", "[preprocessor]")
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
    letterbox_info info{0.75f, 0, 40, {640, 480}};
    cv::Rect2f box{10.f, 50.f, 100.f, 80.f};
    auto result = preprocessor::unletterbox(box, info);

    float exp_x = (10.f - 0.f) / 0.75f / 640.f;
    REQUIRE(result.x == Approx(exp_x).margin(1e-4f));
}

TEST_CASE("Engine constructs from pre-built .engine file", "[engine][integration]")
{
    const char* path = std::getenv("HPCV_TRT_ENGINE");
    if (!path) { WARN("HPCV_TRT_ENGINE not set — skipping."); return; }

    trt_inference_engine::config cfg;
    cfg.engine_path = path;
    cfg.device_id   = 0;

    REQUIRE_NOTHROW([&]{ trt_inference_engine e(cfg); }());
}

TEST_CASE("detect returns < 5 boxes on blank image", "[engine][integration]")
{
    const char* path = std::getenv("HPCV_TRT_ENGINE");
    if (!path) { WARN("HPCV_TRT_ENGINE not set — skipping."); return; }

    trt_inference_engine::config cfg;
    cfg.engine_path = path;
    trt_inference_engine engine(cfg);

    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    auto dets = engine.detect(blank, 0.9f, 0.45f);

    REQUIRE(dets.size() < 5);
}

TEST_CASE("Engine builds from ONNX (requires HPCV_TEST_ONNX)", "[engine][build]")
{
    const char* onnx_path = std::getenv("HPCV_TEST_ONNX");
    if (!onnx_path) { WARN("HPCV_TEST_ONNX not set — skipping."); return; }

    trt_inference_engine::config cfg;
    cfg.onnx_path = onnx_path;
    cfg.fp16      = false;

    REQUIRE_NOTHROW([&]{ trt_inference_engine e(cfg); }());
}
