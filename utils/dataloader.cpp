#include "dataloader.h"
#include <filesystem>
#include <fstream>
#include <random>

CustomDataset::CustomDataset(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& targets)
    : images(images), targets(targets) {
}

torch::data::Example<> CustomDataset::get(size_t index) {
    auto image = images[index].clone();
    auto target = targets[index].clone();

    thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if (dist(rng) < 0.5f) {
        image = image.flip({2});
        if (target.size(0) > 0) {
            target.select(1, 0) = 1.0f - target.select(1, 0);
        }
    }

    return { image, target };
}

torch::optional<size_t> CustomDataset::size() const {
    return images.size();
}

torch::Tensor get_target_data(std::ifstream& file) {
    std::vector<torch::Tensor> lines;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> numbers((std::istream_iterator<float>(iss)), std::istream_iterator<float>());
        if (numbers.empty()) continue;
        std::rotate(numbers.begin(), numbers.begin() + 1, numbers.end());
        numbers.back() += 1.0f;
        lines.push_back(torch::tensor(numbers));
    }

    if (lines.empty()) {
        return torch::zeros({ 0, 5 });
    }

    return torch::stack(lines);
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_data(const std::string& dataset_path, const int& img_size, const std::string& split) {
    std::vector<torch::Tensor> images, labels;
    cv::Mat image;
    std::string path_to_image;

    std::string image_dir = dataset_path + "images/" + split + "/";
    std::string label_dir = dataset_path + "labels/" + split + "/";

    if (!std::filesystem::exists(image_dir)) {
        std::cerr << "Directory not found: " << image_dir << std::endl;
        return { images, labels };
    }

    for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
        path_to_image = entry.path().string();

        image = cv::imread(path_to_image, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load the image " << path_to_image << std::endl;
            continue;
        }

        std::filesystem::path file_label(path_to_image);
        std::ifstream file(label_dir + file_label.stem().string() + ".txt");
        if (!file.is_open()) {
            std::cerr << "Failed to open label file for " << path_to_image << std::endl;
            continue;
        }

        torch::Tensor target = get_target_data(file);
        images.push_back(normalize_image(image, img_size));
        labels.push_back(target);
    }

    return { images, labels };
}
