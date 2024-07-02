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

std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>> get_train_data(const std::string& directory_path, const int& img_size, const torch::Device& device, const int& num_classes) {
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

        images.push_back(normalize_image(image, img_size));
        labels.push_back(get_target_data(file));
    }

    return { images, labels };
}
