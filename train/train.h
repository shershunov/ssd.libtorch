#pragma once
#include "../model/model.h"

enum class PrecisionMode { FP32, FP16, BF16 };

struct GradScaler {
    float scale_;
    float growth_factor_;
    float backoff_factor_;
    float max_scale_;
    int growth_interval_;
    int growth_step_;

    GradScaler(float init_scale = 65536.0f, float growth_factor = 2.0f,
               float backoff_factor = 0.5f, int growth_interval = 2000);

    torch::Tensor scale_loss(const torch::Tensor& loss);
    bool step(torch::optim::Optimizer& optimizer, float max_norm = 0.0f);
};

std::vector<float> train_model(Net& model, torch::Device& device,
    std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& targets,
    std::vector<torch::Tensor>& val_images, std::vector<torch::Tensor>& val_targets,
    const int& num_epochs, const int& batch_size, const float& learning_rate,
    const int& num_workers, PrecisionMode precision, int num_classes,
    const torch::Tensor& anchors);

void load_model(Net& model, const std::string& filename);
