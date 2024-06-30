#pragma once
#include <vector>
#include "utils.h"

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    std::vector<torch::Tensor> images, targets;
    bool shuffle;

public:
    CustomDataset(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& targets);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;
};

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_train_data(const std::string& directory_path, const int& img_size, const torch::Device& device, const int& num_classes);