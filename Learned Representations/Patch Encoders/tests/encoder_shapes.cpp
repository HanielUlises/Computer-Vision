#include <torch/torch.h>
#include <cassert>
#include <iostream>

#include "CNN_Encoder.hpp"

int main() {
    torch::manual_seed(0);

    const int batch = 8;
    const int h = 32;
    const int w = 32;
    const int out_dim = 128;

    CNN_Encoder encoder(out_dim);

    auto x = torch::randn({batch, 1, h, w});
    auto y = encoder->forward(x);

    assert(y.dim() == 2);
    assert(y.size(0) == batch);
    assert(y.size(1) == out_dim);

    std::cout << "encoder_shapes: OK\n";
    return 0;
}
