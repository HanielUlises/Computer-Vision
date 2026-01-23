#include "distance.hpp"
#include <cmath>

float l2_distance(const Embedding& a, const Embedding& b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.dim(); ++i) {
        float d = a.data[i] - b.data[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}

float cosine_distance(const Embedding& a, const Embedding& b) {
    float dot = 0.0f;
    for (size_t i = 0; i < a.dim(); ++i)
        dot += a.data[i] * b.data[i];
    return 1.0f - dot;
}
