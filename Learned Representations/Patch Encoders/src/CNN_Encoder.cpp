#include "CNN_Encoder.hpp"

CNN_EncoderImpl::CNN_EncoderImpl(int out_dim)
    : out_dim_(out_dim) {
    net = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 32, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::MaxPool2d(2),

        torch::nn::Conv2d(torch::nn::Conv2dOptions(64, out_dim_, 3).padding(1)),
        torch::nn::ReLU(),
        torch::nn::AdaptiveAvgPool2d(1),

        torch::nn::Flatten()
    );

    register_module("net", net);
}

torch::Tensor CNN_EncoderImpl::forward(torch::Tensor x) {
    return net->forward(x);
}

int CNN_EncoderImpl::outputDim() const {
    return out_dim_;
}
