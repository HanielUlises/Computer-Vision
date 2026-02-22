#include "descriptor_eval.hpp"

EvalStats evaluate_descriptors(
    torch::Tensor descriptors,
    torch::Tensor labels
) {
    descriptors = descriptors.detach().cpu();
    labels = labels.detach().cpu();

    const int N = descriptors.size(0);

    torch::Tensor dist = torch::cdist(descriptors, descriptors);

    float pos_sum = 0.0f;
    float neg_sum = 0.0f;
    int pos_cnt = 0;
    int neg_cnt = 0;
    int correct_top1 = 0;

    for (int i = 0; i < N; ++i) {
        float best_dist = 1e9f;
        int best_j = -1;

        for (int j = 0; j < N; ++j) {
            if (i == j) continue;

            float d = dist[i][j].item<float>();

            if (labels[i].item<int>() == labels[j].item<int>()) {
                pos_sum += d;
                pos_cnt++;
            } else {
                neg_sum += d;
                neg_cnt++;
            }

            if (d < best_dist) {
                best_dist = d;
                best_j = j;
            }
        }

        if (labels[i].item<int>() == labels[best_j].item<int>()) {
            correct_top1++;
        }
    }

    EvalStats stats;
    stats.mean_pos_dist = pos_sum / pos_cnt;
    stats.mean_neg_dist = neg_sum / neg_cnt;
    stats.top1_accuracy = static_cast<float>(correct_top1) / N;

    return stats;
}
