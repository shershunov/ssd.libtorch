#pragma once
#include <torch/torch.h>

torch::Tensor generate_anchors(int img_size, int num_anchors_per_cell = 2);

torch::Tensor decode_boxes(const torch::Tensor& offsets, const torch::Tensor& anchors);
