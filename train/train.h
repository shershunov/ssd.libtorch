#pragma once
#include "../model/model.h"

std::vector<float> train_model(Net& model, torch::Device& device, std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& targets, const int& num_epochs, const int& batch_size, const float& learning_rate, const int& num_workers);

void load_model(Net& model, const std::string& filename);