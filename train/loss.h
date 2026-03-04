#pragma once
#include <torch/torch.h>

torch::Tensor ssd_loss(
    const torch::Tensor& pred_offsets,
    const torch::Tensor& pred_labels,
    const std::vector<torch::Tensor>& targets,
    const torch::Tensor& anchors);
