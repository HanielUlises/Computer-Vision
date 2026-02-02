#pragma once
#include "Patch_Encoder.hpp"

class CNN_Encoder : public Patch_Encoder {
public:
    CNN_Encoder(int out_dim = 128);

    torch::Tensor forward(torch::Tensor x) override;
    int outputDim() const override;

private:
    int out_dim_;
    torch::nn::Sequential net;
};

TORCH_MODULE(CNN_Encoder);
