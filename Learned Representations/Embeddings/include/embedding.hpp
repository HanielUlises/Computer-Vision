#pragma once
#include <vector>

struct Embedding {
    std::vector<float> data;

    explicit Embedding(size_t dim = 0)
        : data(dim, 0.0f) {}

    size_t dim() const { return data.size(); }

    float norm() const;
    void normalize();
};
