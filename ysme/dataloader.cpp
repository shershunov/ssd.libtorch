#include "dataloader.h"
#include <filesystem>
#include <fstream>
#include <random>

CustomDataset::CustomDataset(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& targets, const bool& shuffle) 
    : images(images), targets(targets) {
    if (shuffle) {
        shuffleDataset();
    }
}

torch::data::Example<> CustomDataset::get(size_t index) {
    return { images[index], targets[index] };
}

torch::optional<size_t> CustomDataset::size() const {
    return images.size();
}

void CustomDataset::shuffleDataset() {
    std::vector<size_t> indices(this->images.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<torch::Tensor> shuffledImages(this->images.size());
    std::vector<torch::Tensor> shuffledTargets(this->targets.size());

    for (int i = 0; i < indices.size(); ++i) {
        size_t index = indices[i];
        shuffledImages[i] = this->images[index];
        shuffledTargets[i] = this->targets[index];
    }

    this->images.swap(shuffledImages);
    this->targets.swap(shuffledTargets);
}

torch::Tensor get_target_data(std::ifstream& file) {
    std::vector<torch::Tensor> lines;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<float> numbers((std::istream_iterator<float>(iss)), std::istream_iterator<float>());
        std::rotate(numbers.begin(), numbers.begin() + 1, numbers.end());
        lines.push_back(torch::tensor(numbers));
    }

    return torch::stack(lines);
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_train_data(const std::string& directory_path, const int& resize_width, const int& resize_height, const torch::Device& device, const int& num_classes) {
    std::vector<torch::Tensor> images, labels;
    cv::Mat image; 
    std::string path;

    const std::string image_dir = directory_path + "images/train/";
    const std::string label_dir = directory_path + "labels/train/";

    for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
        path = entry.path().string();

        image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Corrupt image " << path << std::endl;
            continue;
        }

        std::filesystem::path file_label(path);
        std::ifstream file(label_dir + file_label.stem().string() + ".txt");
        if (!file.is_open()) {
            std::cerr << "Failed to open label file for " << path << std::endl;
            continue;
        }

        images.push_back(normalize_image(image, resize_width, resize_height, device));
        labels.push_back(get_target_data(file));
    }

    return { images, labels };
}
