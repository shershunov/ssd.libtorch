#include "test.h"
#include "anchors.h"
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

std::vector<int> tensor_to_int_bbox(const torch::Tensor& input, int img_width, int img_height) {
    auto cpu_input = input.cpu().contiguous();
    auto accessor = cpu_input.accessor<float, 2>();
    int boxes = accessor.size(0);

    std::vector<int> result;
    result.reserve(boxes * 4);

    for (int i = 0; i < boxes; ++i) {
        result.push_back(static_cast<int>(accessor[i][0] * img_width));
        result.push_back(static_cast<int>(accessor[i][1] * img_height));
        result.push_back(static_cast<int>(accessor[i][2] * img_width));
        result.push_back(static_cast<int>(accessor[i][3] * img_height));
    }

    return result;
}

std::unordered_map<int, std::string> get_labels(const std::string& path) {
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

void test_model(Net& model, torch::Device& device, int img_size, std::string dataset_path, const torch::Tensor& anchors) {
    torch::NoGradGuard no_grad;
    model.eval();
    int a = 0;
    std::string input_dir = dataset_path + "images/train/";
    std::string output_dir = dataset_path + "test/";

    if (!std::filesystem::exists(output_dir)) {
        std::filesystem::create_directory(output_dir);
    }

    std::unordered_map<int, std::string> labels = get_labels(dataset_path + "labels.txt");
    auto anchors_dev = anchors.to(device);

    for (const auto& entry : std::filesystem::directory_iterator(input_dir)) {
        auto path = entry.path().string();
        ++a;

        cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to load the image " << path << std::endl;
            continue;
        }
        cv::Mat image_with_boxes = image.clone();

        torch::Tensor test_img = normalize_image(image, img_size).to(device).unsqueeze(0);

        auto [offsets, scores] = model.forward(test_img);

        auto pred_offsets = offsets.squeeze(0);
        auto pred_scores = scores.squeeze(0);

        auto pred_boxes = decode_boxes(pred_offsets, anchors_dev);
        auto probs = torch::softmax(pred_scores, 1);

        auto fg_probs = probs.slice(1, 1);
        auto [max_prob, max_cls] = fg_probs.max(1);

        auto conf_mask = max_prob > 0.25f;

        if (conf_mask.sum().item<int>() == 0) {
            cv::imwrite(output_dir + std::to_string(a) + ".png", image_with_boxes);
            continue;
        }

        auto filtered_boxes = pred_boxes.index({ conf_mask });
        auto filtered_probs = fg_probs.index({ conf_mask });
        auto filtered_cls = max_cls.index({ conf_mask });

        torch::Tensor boxes_xyxy = xywh_to_xyxy(filtered_boxes);
        auto keep_indices = non_max_suppression(boxes_xyxy, filtered_probs, 0.7);
        torch::Tensor keep_indices_tensor = torch::tensor(keep_indices).to(device);

        torch::Tensor kept_boxes = boxes_xyxy.index_select(0, keep_indices_tensor);
        torch::Tensor kept_cls = filtered_cls.index_select(0, keep_indices_tensor);

        auto boxes_int = tensor_to_int_bbox(kept_boxes, image_with_boxes.cols, image_with_boxes.rows);
        auto cls_cpu = kept_cls.cpu();
        auto cls_accessor = cls_cpu.accessor<int64_t, 1>();

        for (int i = 0; i < (int)cls_cpu.size(0); ++i) {
            int box[4] = { boxes_int[i * 4], boxes_int[i * 4 + 1], boxes_int[i * 4 + 2], boxes_int[i * 4 + 3] };
            plot_box(box, image_with_boxes, labels[static_cast<int>(cls_accessor[i])], 2);
        }

        cv::imwrite(output_dir + std::to_string(a) + ".png", image_with_boxes);
    }
}
