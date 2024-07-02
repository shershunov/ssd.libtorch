#include "../utils/dataloader.h"
#include "loss.h"
#include "train.h"
#include <cuda_runtime.h>
#include <fstream>
#include <vector>
#include <windows.h>

std::tuple<torch::Tensor, std::vector<torch::Tensor>> collate_fn(std::vector<torch::data::Example<torch::Tensor, torch::Tensor>>& batch, torch::Device& device) {
    std::vector<torch::Tensor> images;
    std::vector<torch::Tensor> targets;

    for (const auto& example : batch) {
        images.push_back(example.data);
        targets.push_back(example.target);
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

std::vector<float> train_model(Net& model, torch::Device& device, std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& targets,
            const int& num_epochs, const int& batch_size, const float& learning_rate, const int& num_workers) {
    torch::Tensor loss;
    size_t free_mem, total_mem;
    float used_mem, epoch_loss;
    std::vector<float> losses_train;
    torch::optim::AdamW optimizer = torch::optim::AdamW(model.parameters(), learning_rate);

    model.train();

    auto custom_dataset = CustomDataset(images, targets);
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        custom_dataset,
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(num_workers)
    );
    float best_loss = std::numeric_limits<float>::infinity();

    int num_batches = (images.size() - 1) / batch_size + 1;
    std::cout << "Num batches: " << num_batches << std::endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        epoch_loss = 0;

        if (device.is_cuda()) {
            cudaMemGetInfo(&free_mem, &total_mem);
            used_mem = (static_cast<float>(total_mem) - static_cast<float>(free_mem)) / (1024 * 1024 * 1024);
            std::cout << "GPU_mem: " << " Epoch: " << std::endl;
            std::cout << "  " << std::round(used_mem * 100) / 100 << "    " << epoch + 1 << '/' << num_epochs << std::endl;
        }

        for (auto& batch : *data_loader) {
            optimizer.zero_grad();

            auto [images, targets] = collate_fn(batch, device);
            auto [boxes, scores] = model.forward(images);

            loss = ssd_loss(boxes, scores, targets);
            epoch_loss += loss.item<float>();

            loss.backward();
            optimizer.step();
        }

        epoch_loss /= num_batches;
        losses_train.push_back(epoch_loss);

        if (epoch_loss < best_loss && epoch != 0) {
            best_loss = epoch_loss;
            SetConsoleTitle(("Best loss: " + std::to_string(best_loss) + " on " + std::to_string(epoch + 1) + " epoch").c_str());
            save_model(model, "best.pt");
        }
        std::cout << " Loss: " << std::round(epoch_loss * 10000) / 10000 << std::endl << std::endl;
    }
    std::cout << "Saved best model with loss: " << best_loss;
    save_model(model, "last.pt");
    return losses_train;
}
