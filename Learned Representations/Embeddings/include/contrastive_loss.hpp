#pragma once
#include "embedding.hpp"

float contrastive_loss(
    const Embedding& a,
    const Embedding& b,
    bool same_class,
    float margin = 1.0f
);
