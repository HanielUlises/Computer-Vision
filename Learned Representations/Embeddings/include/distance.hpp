#pragma once
#include "embedding.hpp"

float l2_distance(const Embedding& a, const Embedding& b);
float cosine_distance(const Embedding& a, const Embedding& b);
