#pragma once
#include <torch/torch.h>

struct EvalStats {
    float mean_pos_dist;
    float mean_neg_dist;
    float top1_accuracy;
};

EvalStats evaluate_descriptors(
    torch::Tensor descriptors,
    torch::Tensor labels
);
