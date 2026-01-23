#pragma once
#include <torch/torch.h>

// Standard triplet loss for metric learning.
// Enforces d(anchor, positive) + margin < d(anchor, negative)
inline torch::Tensor triplet_loss(
    const torch::Tensor& anchor,
    const torch::Tensor& positive,
    const torch::Tensor& negative,
    float margin = 1.0f
) {
    auto d_ap = torch::norm(anchor - positive, 2, 1);
    auto d_an = torch::norm(anchor - negative, 2, 1);
    return torch::mean(torch::relu(d_ap - d_an + margin));
}
