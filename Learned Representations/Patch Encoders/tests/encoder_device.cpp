#include <torch/torch.h>
#include <cassert>
#include <iostream>

#include "CNN_Encoder.hpp"

int main() {
    CNN_Encoder encoder(64);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
    }

    encoder->to(device);

    auto x = torch::randn({4, 1, 32, 32}).to(device);
    auto y = encoder->forward(x);

    assert(y.device().type() == device.type());
    assert(y.size(1) == 64);

    std::cout << "encoder_device: OK on "
              << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    return 0;
}
