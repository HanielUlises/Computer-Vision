#pragma once
#include <torch/torch.h>

class Patch_Encoder : public torch::nn::Module {
public:
    virtual torch::Tensor forward(torch::Tensor patch) = 0;
    virtual int outputDim() const = 0;
    virtual ~Patch_Encoder() = default;
};
