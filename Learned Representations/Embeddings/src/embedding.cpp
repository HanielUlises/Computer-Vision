#include "embedding.hpp"
#include <cmath>

float Embedding::norm() const {
    float sum = 0.0f;
    for (float v : data)
        sum += v * v;
    return std::sqrt(sum);
}

void Embedding::normalize() {
    float n = norm();
    if (n < 1e-8f) return;
    for (float& v : data)
        v /= n;
}
