#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include "torch_inference_engine.hpp"
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

TEST_CASE("to_tensor shape and values", "[preprocessor]")
{
    cv::Mat rgb(224, 224, CV_8UC3, cv::Scalar(255, 128, 0));
    auto t = preprocessor::to_tensor(rgb, norm_zero_one());

    REQUIRE(t.dim() == 4);
    REQUIRE(t.size(0) == 1);
    REQUIRE(t.size(1) == 3);
    REQUIRE(t.size(2) == 224);
    REQUIRE(t.size(3) == 224);

    // Red channel = 255/255 = 1.0
    REQUIRE(t[0][0][0][0].item<float>() == Approx(1.0f).margin(1e-4f));
    // Green channel = 128/255
    REQUIRE(t[0][1][0][0].item<float>() == Approx(128.f/255.f).margin(1e-4f));
}

TEST_CASE("to_batch_tensor: shape [N,3,H,W]", "[preprocessor]")
{
    std::vector<cv::Mat> imgs = {
        cv::Mat(64, 64, CV_8UC3, cv::Scalar(255, 0, 0)),
        cv::Mat(64, 64, CV_8UC3, cv::Scalar(0, 255, 0)),
    };
    auto t = preprocessor::to_batch_tensor(imgs, norm_zero_one());

    REQUIRE(t.size(0) == 2);
    REQUIRE(t.size(1) == 3);
    REQUIRE(t.size(2) == 64);
    REQUIRE(t.size(3) == 64);
}

TEST_CASE("unletterbox round-trips bbox", "[preprocessor]")
{
    letterbox_info info{0.75f, 0, 40, {640, 480}};
    cv::Rect2f box{10.f, 50.f, 100.f, 80.f};
    auto result = preprocessor::unletterbox(box, info);

    float exp_x = (10.f) / 0.75f / 640.f;
    REQUIRE(result.x == Approx(exp_x).margin(1e-4f));
}

// ── Engine integration tests ─────────────────────────────────────────────────

TEST_CASE("Engine constructs on CPU with a TorchScript model", "[engine][integration]")
{
    const char* path = std::getenv("HPCV_TORCH_MODEL");
    if (!path) { WARN("HPCV_TORCH_MODEL not set — skipping."); return; }

    torch_inference_engine::config cfg;
    cfg.model_path = path;
    cfg.device     = torch_inference_engine::config::device_type::cpu;

    REQUIRE_NOTHROW([&]{ torch_inference_engine e(cfg); }());
}

TEST_CASE("detect returns < 5 boxes on blank image (CPU)", "[engine][integration]")
{
    const char* path = std::getenv("HPCV_TORCH_MODEL");
    if (!path) { WARN("HPCV_TORCH_MODEL not set — skipping."); return; }

    torch_inference_engine::config cfg;
    cfg.model_path = path;
    cfg.device     = torch_inference_engine::config::device_type::cpu;
    torch_inference_engine engine(cfg);

    cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    auto dets = engine.detect(blank, 0.9f, 0.45f);

    REQUIRE(dets.size() < 5);
}

TEST_CASE("raw run returns a tensor IValue", "[engine][integration]")
{
    const char* path = std::getenv("HPCV_TORCH_MODEL");
    if (!path) { WARN("HPCV_TORCH_MODEL not set — skipping."); return; }

    torch_inference_engine::config cfg;
    cfg.model_path = path;
    cfg.device     = torch_inference_engine::config::device_type::cpu;
    torch_inference_engine engine(cfg);

    auto t   = torch::zeros({1, 3, 640, 640});
    auto out = engine.run(t);

    REQUIRE(!out.empty());
    REQUIRE(out[0].isTensor());
}
