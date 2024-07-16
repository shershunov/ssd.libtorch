#include "dataloader.h"
#include <filesystem>
#include <fstream>
#include <random>

CustomDataset::CustomDataset(std::vector<torch::Tensor>& images, std::vector<torch::Tensor>& targets) 
    : images(images), targets(targets) {
}

torch::data::Example<> CustomDataset::get(size_t index) {
    return { images[index], targets[index] };
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
        std::rotate(numbers.begin(), numbers.begin() + 1, numbers.end());
        lines.push_back(torch::tensor(numbers));
    }

    return torch::stack(lines);
}

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_train_data(const std::string& dataset_path, const int& img_size, const torch::Device& device, const int& num_classes) {
    std::vector<torch::Tensor> images, labels;
    cv::Mat image; 
    std::string path_to_image;

    std::string image_dir = dataset_path + "images/train/";
    std::string label_dir = dataset_path + "labels/train/";

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

        images.push_back(normalize_image(image, img_size));
        labels.push_back(get_target_data(file));
    }

    return { images, labels };
}
