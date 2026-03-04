#pragma once
#include <vector>
#include "../model/model.h"

struct MapResult {
    float map_50;
    float map_50_95;
};

MapResult validate(Net& model, torch::Device& device,
                   std::vector<torch::Tensor>& images,
                   std::vector<torch::Tensor>& targets,
                   int num_classes, int batch_size,
                   const torch::Tensor& anchors);
