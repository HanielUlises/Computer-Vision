#include <torch/torch.h>
#include "patch_net.hpp"
#include "patch_dataset.hpp"
#include "triplet_loss.hpp"
#include <iostream>

int main() {
    torch::manual_seed(0);

    PatchDataset dataset("dataset/");
    PatchNet net;
    net->train();

    torch::optim::Adam optimizer(net->parameters(), 1e-3);

    const int iterations = 10000;
    const int batch_size = 16;

    for (int it = 0; it < iterations; ++it) {
        std::vector<torch::Tensor> A, P, N;

        for (int i = 0; i < batch_size; ++i) {
            auto a = dataset.get_random();
            auto p = dataset.get_same(a.label);
            auto n = dataset.get_diff(a.label);

            A.push_back(a.patch);
            P.push_back(p.patch);
            N.push_back(n.patch);
        }

        auto tA = torch::stack(A);
        auto tP = torch::stack(P);
        auto tN = torch::stack(N);

        auto fA = net->forward(tA);
        auto fP = net->forward(tP);
        auto fN = net->forward(tN);

        auto loss = triplet_loss(fA, fP, fN);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        if (it % 500 == 0) {
            std::cout << "Iter " << it
                      << " | Loss: " << loss.item<float>()
                      << std::endl;
        }
    }

    torch::save(net, "models/patch_net.pt");
    return 0;
}
