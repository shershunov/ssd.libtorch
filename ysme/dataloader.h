#pragma once
#include <vector>
#include "utils.h"

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    std::vector<torch::Tensor> images, targets;
    bool shuffle;

public:
    CustomDataset(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& targets, const bool& shuffle);

    torch::data::Example<> get(size_t index) override;

    torch::optional<size_t> size() const override;

    void shuffleDataset();
};

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_train_data(const std::string& directory_path, const int& resize_width, const int& resize_height, const torch::Device& device, const int& num_classes);