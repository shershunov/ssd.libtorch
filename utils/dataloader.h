#pragma once
#include <vector>
#include "utils.h"

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    std::vector<torch::Tensor> images, targets;

public:
    CustomDataset(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& targets);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;
};

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_data(const std::string& dataset_path, const int& img_size, const std::string& split);