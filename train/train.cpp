#include "../utils/dataloader.h"
#include "../utils/metrics.h"
#include "loss.h"
#include "train.h"
#include <ATen/autocast_mode.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>
#define NOMINMAX
#include <windows.h>

GradScaler::GradScaler(float init_scale, float growth_factor, float backoff_factor, int growth_interval)
    : scale_(init_scale), growth_factor_(growth_factor), backoff_factor_(backoff_factor),
      max_scale_(65536.0f * 128.0f), growth_interval_(growth_interval), growth_step_(0) {}

torch::Tensor GradScaler::scale_loss(const torch::Tensor& loss) {
    return loss * scale_;
}

bool GradScaler::step(torch::optim::Optimizer& optimizer, float max_norm) {
    std::vector<torch::Tensor> all_params;
    std::vector<torch::Tensor> flat_grads;

    for (auto& group : optimizer.param_groups()) {
        for (auto& param : group.params()) {
            if (!param.grad().defined()) continue;
            param.grad().div_(scale_);
            flat_grads.push_back(param.grad().view(-1));
            all_params.push_back(param);
        }
    }

    if (!flat_grads.empty()) {
        auto all_grads = torch::cat(flat_grads);
        if (!torch::isfinite(all_grads).all().item<bool>()) {
            optimizer.zero_grad();
            scale_ *= backoff_factor_;
            growth_step_ = 0;
            return false;
        }
    }

    if (max_norm > 0 && !all_params.empty()) {
        torch::nn::utils::clip_grad_norm_(all_params, max_norm);
    }

    optimizer.step();
    growth_step_++;
    if (growth_step_ >= growth_interval_) {
        scale_ = std::min(scale_ * growth_factor_, max_scale_);
        growth_step_ = 0;
    }
    return true;
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> collate_fn(std::vector<torch::data::Example<torch::Tensor, torch::Tensor>>& batch, torch::Device& device) {
    std::vector<torch::Tensor> images;
    std::vector<torch::Tensor> targets;
    images.reserve(batch.size());
    targets.reserve(batch.size());

    for (const auto& example : batch) {
        images.push_back(example.data);
        targets.push_back(example.target.to(device));
    }

    return { torch::stack(images, 0).to(device), targets };
}

void save_model(const Net& model, const std::string& filename) {
    try {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        torch::serialize::OutputArchive archive;
        model.save(archive);
        archive.save_to(file);
    }
    catch (const std::exception& e) {
        std::cerr << "Error saving model: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown error occurred while saving model" << std::endl;
    }
}

void load_model(Net& model, const std::string& filename) {
    try {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        torch::serialize::InputArchive archive;
        archive.load_from(file);
        model.load(archive);
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "Unknown error occurred while loading model" << std::endl;
    }
}

std::vector<float> train_model(Net& model, torch::Device& device,
            std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& targets,
            std::vector<torch::Tensor>& val_images, std::vector<torch::Tensor>& val_targets,
            const int& num_epochs, const int& batch_size, const float& learning_rate,
            const int& num_workers, PrecisionMode precision, int num_classes,
            const torch::Tensor& anchors) {
    torch::Tensor loss;
    size_t free_mem, total_mem;
    float used_mem, epoch_loss;
    std::vector<float> losses_train;
    torch::optim::AdamW optimizer = torch::optim::AdamW(model.parameters(), learning_rate);

    const float max_norm = 1.0f;
    constexpr int warmup_epochs = 5;
    float lr_min = learning_rate * 0.01f;
    float lr_max = learning_rate;

    bool use_amp = (precision != PrecisionMode::FP32) && device.is_cuda();
    bool use_scaler = (precision == PrecisionMode::FP16) && device.is_cuda();
    at::ScalarType amp_dtype = (precision == PrecisionMode::FP16) ? at::kHalf : at::kBFloat16;
    at::DeviceType device_type = device.is_cuda() ? at::kCUDA : at::kCPU;

    GradScaler scaler;

    if (use_amp) {
        std::string dtype_str = (precision == PrecisionMode::FP16) ? "FP16" : "BF16";
        std::cout << "AMP enabled: " << dtype_str;
        if (use_scaler) std::cout << " + GradScaler";
        std::cout << std::endl;
    } else {
        std::cout << "Precision: FP32" << std::endl;
    }

    std::cout << "Grad norm clipping: " << max_norm << std::endl;

    model.train();

    auto custom_dataset = CustomDataset(images, targets);
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        custom_dataset,
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers)
    );

    float best_map = 0.0f;
    int best_epoch = 0;

    int num_batches = static_cast<int>((images.size() + batch_size - 1) / batch_size);
    const int bar_width = 30;

    std::cout << "Batches: " << num_batches << "  Val images: " << val_images.size() << std::endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float lr;
        if (epoch < warmup_epochs) {
            lr = lr_min + (lr_max - lr_min) * static_cast<float>(epoch + 1) / warmup_epochs;
        } else {
            float progress = static_cast<float>(epoch - warmup_epochs) / static_cast<float>(std::max(num_epochs - warmup_epochs, 1));
            lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + std::cos(M_PI * progress));
        }
        for (auto& group : optimizer.param_groups()) {
            static_cast<torch::optim::AdamWOptions&>(group.options()).lr(lr);
        }

        model.train();
        torch::Tensor epoch_loss_acc = torch::zeros({ 1 }, torch::TensorOptions().device(device));

        if (device.is_cuda()) {
            cudaMemGetInfo(&free_mem, &total_mem);
            used_mem = (static_cast<float>(total_mem) - static_cast<float>(free_mem)) / (1024 * 1024 * 1024);
        }

        int batch_idx = 0;
        for (auto& batch : *data_loader) {
            optimizer.zero_grad();

            auto [images, targets] = collate_fn(batch, device);

            if (use_amp) {
                at::autocast::set_autocast_enabled(device_type, true);
                at::autocast::set_autocast_dtype(device_type, amp_dtype);
                at::autocast::increment_nesting();
            }

            auto [boxes, scores] = model.forward(images);
            loss = ssd_loss(boxes, scores, targets, anchors);

            if (use_amp) {
                at::autocast::decrement_nesting();
                at::autocast::set_autocast_enabled(device_type, false);
                at::autocast::clear_cache();
            }

            epoch_loss_acc += loss.detach();

            if (use_scaler) {
                scaler.scale_loss(loss).backward();
                scaler.step(optimizer, max_norm);
            } else {
                loss.backward();
                torch::nn::utils::clip_grad_norm_(model.parameters(), max_norm);
                optimizer.step();
            }

            ++batch_idx;
            int filled = bar_width * batch_idx / num_batches;
            float running_loss = epoch_loss_acc.item<float>() / batch_idx;

            std::cout << "\r  Epoch " << epoch + 1 << "/" << num_epochs;
            if (device.is_cuda()) {
                std::cout << "  GPU: " << std::fixed << std::setprecision(2) << used_mem << "GB";
            }
            std::cout << "  [";
            for (int b = 0; b < bar_width; ++b) std::cout << (b < filled ? '#' : '.');
            std::cout << "] " << batch_idx << "/" << num_batches
                      << "  Loss: " << std::fixed << std::setprecision(4) << running_loss
                      << "   " << std::flush;
        }

        epoch_loss = epoch_loss_acc.item<float>() / num_batches;
        losses_train.push_back(epoch_loss);

        std::cout << "\r  Epoch " << epoch + 1 << "/" << num_epochs;
        if (device.is_cuda()) {
            std::cout << "  GPU: " << std::fixed << std::setprecision(2) << used_mem << "GB";
        }
        std::cout << "  [";
        for (int b = 0; b < bar_width; ++b) std::cout << '#';
        std::cout << "] " << num_batches << "/" << num_batches
                  << "  Loss: " << std::fixed << std::setprecision(4) << epoch_loss
                  << "  LR: " << std::scientific << std::setprecision(2) << lr
                  << "   " << std::endl;

        if (!val_images.empty()) {
            auto [map_50, map_50_95] = validate(model, device, val_images, val_targets, num_classes, batch_size, anchors);

            std::cout << "  mAP@0.5: " << std::fixed << std::setprecision(4) << map_50
                      << "  mAP@0.5:0.95: " << map_50_95;

            if (map_50_95 > best_map) {
                best_map = map_50_95;
                best_epoch = epoch + 1;
                SetConsoleTitle(("Best mAP@0.5:0.95: " + std::to_string(best_map) + " epoch " + std::to_string(best_epoch)).c_str());
                save_model(model, "best.pt");
                std::cout << " [BEST]";
            }

            std::cout << std::endl;
        }

        std::cout << std::endl;
    }

    std::cout << "Best mAP@0.5:0.95: " << best_map << " at epoch " << best_epoch << std::endl;
    save_model(model, "last.pt");
    return losses_train;
}
