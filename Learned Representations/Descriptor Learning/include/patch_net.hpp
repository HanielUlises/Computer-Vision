#pragma once
#include <torch/torch.h>

// CNN for learned local descriptor extraction.
// Maps a 32x32 grayscale image patch to a 128-D L2-normalized embedding.
// Optimized for metric learning (triplet / contrastive losses) and intended
// as a drop-in replacement for hand-crafted descriptors such as SIFT or HOG.
struct PatchNetImpl : torch::nn::Module {
    torch::nn::Sequential stem;
    torch::nn::Sequential block1;
    torch::nn::Sequential block2;
    torch::nn::Conv2d projection{nullptr};

    PatchNetImpl() {
        stem = register_module("stem", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1).bias(false)),
            torch::nn::BatchNorm2d(32),
            torch::nn::ReLU(true)
        ));

        block1 = register_module("block1", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 32, 3)
                .stride(2).padding(1).groups(32).bias(false)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 1).bias(false)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(true)
        ));

        block2 = register_module("block2", torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3)
                .stride(2).padding(1).groups(64).bias(false)),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 1).bias(false)),
            torch::nn::BatchNorm2d(128),
            torch::nn::ReLU(true)
        ));

        projection = register_module(
            "projection",
            torch::nn::Conv2d(128, 128, 1)
        );
    }

    torch::Tensor forward(torch::Tensor x) {
        x = stem->forward(x);
        x = block1->forward(x);
        x = block2->forward(x);
        x = projection->forward(x);
        x = torch::mean(x, {2, 3});
        return torch::nn::functional::normalize(x);
    }
};

TORCH_MODULE(PatchNet);
