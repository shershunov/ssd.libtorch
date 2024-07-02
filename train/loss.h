#pragma once
#include <torch/torch.h>

torch::Tensor ssd_loss(
    const torch::Tensor& pred_boxes,
    const torch::Tensor& pred_classes,
    const std::vector<torch::Tensor>& targets);
