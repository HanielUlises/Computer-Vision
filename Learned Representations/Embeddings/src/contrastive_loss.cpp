#include "contrastive_loss.hpp"
#include "distance.hpp"
#include <algorithm>

float contrastive_loss(
    const Embedding& a,
    const Embedding& b,
    bool same_class,
    float margin
) {
    float d = l2_distance(a, b);

    if (same_class) {
        return d * d;
    } else {
        float m = std::max(0.0f, margin - d);
        return m * m;
    }
}
