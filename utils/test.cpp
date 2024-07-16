#include "test.h"
#include "utils.h"
#include <filesystem>
#include <fstream>
#include <unordered_map>

void plot_box(const int x[4], cv::Mat& img, const std::string& label, int line_thickness) {
    int tl = line_thickness == 0 ? static_cast<int>(.002 * (img.rows + img.cols) / 2 + 1) : line_thickness;
    cv::Point c1(x[0], x[1]);
    cv::Point c2(x[2], x[3]);
    cv::rectangle(img, c1, c2, cv::Scalar(0, 165, 255), tl, cv::LINE_AA);

    c2 = cv::Point(c1.x + static_cast<int>(label.length() * 10.8), c1.y - 18);
    cv::rectangle(img, cv::Point(c1.x - 1, c1.y), c2, cv::Scalar(0, 165, 255), cv::FILLED, cv::LINE_AA);

    cv::putText(img, label, cv::Point(c1.x + 2, c1.y - 4), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(225, 255, 255), 1, cv::LINE_AA);
}

std::vector<int> tensor_to_int_bbox(torch::Tensor& input, int img_size) {
    int boxes = input.size(0);

    std::vector<int> result;
    result.reserve(boxes * 4);

    for (int i = 0; i < boxes; ++i) {
        int x1 = static_cast<int>((input[i][0].item<float>() * img_size));
        int y1 = static_cast<int>((input[i][1].item<float>() * img_size));
        int x2 = static_cast<int>((input[i][2].item<float>() * img_size));
        int y2 = static_cast<int>((input[i][3].item<float>() * img_size));

        result.push_back(x1);
        result.push_back(y1);
        result.push_back(x2);
        result.push_back(y2);
    }

    return result;
}

std::unordered_map<int, std::string> get_labels(std::string path) {
    std::unordered_map<int, std::string> labels;
    std::string line;

    std::ifstream file(path);
    while (std::getline(file, line)) {
        size_t delimiter_pos = line.find(':');
        if (delimiter_pos != std::string::npos) {
            int key = std::stoi(line.substr(0, delimiter_pos));
            std::string value = line.substr(delimiter_pos + 1);

            size_t start = value.find_first_not_of(" \t");
            if (start != std::string::npos) {
                value = value.substr(start);
            }

            labels[key] = value;
        }
    }

    file.close();
    return labels;  
}

void test_model(Net& model, torch::Device& device, int img_size, std::string dataset_path) {
    model.eval();
    int a = 0;
    std::string input_dir = dataset_path + "images/train/";
    std::string output_dir = dataset_path + "test/";

    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directory(output_dir);
    }

    std::unordered_map<int, std::string> labels = get_labels(dataset_path + "labels.txt");

    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        auto path = entry.path().string();
        ++a;

        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        cv::Mat image_with_boxes = image.clone();
        if (image.empty()) {
            std::cerr << "Failed to load the image " << path << std::endl;
            continue;
        }

        torch::Tensor test_img = normalize_image(image, img_size).to(device).unsqueeze(0);

        auto [boxes, scores] = model.forward(test_img);

        torch::Tensor boxes_xyxy = xywh_to_xyxy(boxes.squeeze(0));
        auto keep_indices = non_max_suppression(boxes_xyxy, scores.squeeze(0), 0.7);
        torch::Tensor keep_indices_tensor = torch::tensor(keep_indices).to(device);

        torch::Tensor kept_boxes = boxes_xyxy.index_select(0, keep_indices_tensor);
        torch::Tensor kept_scores = scores.squeeze(0).index_select(0, keep_indices_tensor);

        auto boxes_int = tensor_to_int_bbox(kept_boxes, img_size);
        scores = scores.squeeze(0);
        for (int i = 0; i < boxes_int.size(); i += 4) {
            int box[4];
            box[0] = boxes_int[i];
            box[1] = boxes_int[i + 1];
            box[2] = boxes_int[i + 2];
            box[3] = boxes_int[i + 3];

            auto current_scores = scores[i];
            auto index_class = current_scores.max(0);
            int64_t max_index = std::get<1>(index_class).item<int64_t>();

            plot_box(box, image_with_boxes, labels[max_index], 2);
        }

        bool result = cv::imwrite(output_dir + std::to_string(a) + ".png", image_with_boxes);
    }
}
