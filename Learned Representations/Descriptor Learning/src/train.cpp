#include <torch/torch.h>
#include <iostream>
#include <filesystem>

#include "patch_net.hpp"
#include "patch_dataset.hpp"
#include "triplet_loss.hpp"

int main(int argc, char** argv) {
    torch::manual_seed(0);

    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        device = torch::kCUDA;
    }

    PatchDataset dataset("dataset/");

    PatchNet net;
    net->to(device);
    net->train();

    torch::optim::Adam optimizer(
        net->parameters(),
        torch::optim::AdamOptions(1e-3)
    );

    const int iterations = 10000;
    const int batch_size = 16;

    std::filesystem::create_directories("models");

    for (int it = 0; it < iterations; ++it) {
        std::vector<torch::Tensor> A, P, N;
        A.reserve(batch_size);
        P.reserve(batch_size);
        N.reserve(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            auto a = dataset.get_random();
            auto p = dataset.get_same(a.label);
            auto n = dataset.get_diff(a.label);

            A.push_back(a.patch);
            P.push_back(p.patch);
            N.push_back(n.patch);
        }

        auto tA = torch::stack(A).to(device);
        auto tP = torch::stack(P).to(device);
        auto tN = torch::stack(N).to(device);

        auto fA = net->forward(tA);
        auto fP = net->forward(tP);
        auto fN = net->forward(tN);

        auto loss = triplet_loss(fA, fP, fN);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (it % 500 == 0) {
            std::cout
                << "[iter " << it << "] "
                << "loss = " << loss.item<float>()
                << std::endl;
        }
    }

    net->eval();
    torch::save(net, "models/patch_net.pt");

    return 0;
}
