#include "utils.h"

torch::Tensor normalize_image(cv::Mat& image, const int& img_size) {
    cv::resize(image, image, cv::Size(img_size, img_size));
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    torch::Tensor tensor_image = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kFloat32).permute({ 2, 0, 1 });
    tensor_image = torch::data::transforms::Normalize<>({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 })(tensor_image);

    return tensor_image;
}

torch::Tensor xywh_to_xyxy(const torch::Tensor& xywh) {
    torch::autograd::variable_list xywh_slices = xywh.split(1, 1);
    torch::Tensor x = xywh_slices[0];
    torch::Tensor y = xywh_slices[1];

    torch::Tensor half_w = xywh_slices[2] / 2;
    torch::Tensor half_h = xywh_slices[3] / 2;

    torch::Tensor x1 = x - half_w;
    torch::Tensor y1 = y - half_h;
    torch::Tensor x2 = x + half_w;
    torch::Tensor y2 = y + half_h;

    return torch::cat({ x1, y1, x2, y2 }, 1);
}

float IoU(const torch::Tensor& box1, const torch::Tensor& box2) {
    auto x1 = std::max(box1[0].item<float>(), box2[0].item<float>());
    auto y1 = std::max(box1[1].item<float>(), box2[1].item<float>());
    auto x2 = std::min(box1[2].item<float>(), box2[2].item<float>());
    auto y2 = std::min(box1[3].item<float>(), box2[3].item<float>());

    float inter_area = std::max(0.0f, x2 - x1 + 1) * std::max(0.0f, y2 - y1 + 1);
    float box1_area = (box1[2].item<float>() - box1[0].item<float>() + 1) * (box1[3].item<float>() - box1[1].item<float>() + 1);
    float box2_area = (box2[2].item<float>() - box2[0].item<float>() + 1) * (box2[3].item<float>() - box2[1].item<float>() + 1);

    return inter_area / (box1_area + box2_area - inter_area);
}

std::vector<int> non_max_suppression(const torch::Tensor& boxes, const torch::Tensor& scores, float iou_threshold = 0.5) {
    assert(boxes.size(0) == scores.size(0));
    int num_boxes = boxes.size(0);

    auto max_scores = std::get<0>(scores.max(1));
    auto max_classes = std::get<1>(scores.max(1));
    auto device = boxes.device();

    auto indices = torch::arange(num_boxes, torch::kInt64);

    torch::Tensor sorted_indices = std::get<1>(max_scores.sort(0, true));

    std::vector<int> keep;

    while (sorted_indices.size(0) > 0) {
        int current = sorted_indices[0].item<int>();
        keep.push_back(current);

        if (sorted_indices.size(0) == 1) {
            break;
        }

        sorted_indices = sorted_indices.slice(0, 1);

        auto remaining_boxes = boxes.index_select(0, sorted_indices);

        auto ious = torch::empty(sorted_indices.size(0));
        for (int i = 0; i < sorted_indices.size(0); ++i) {
            ious[i] = IoU(boxes[current], remaining_boxes[i]);
        }

        auto mask = ious <= iou_threshold;
        sorted_indices = sorted_indices.masked_select(mask.to(device));
    }

    return keep;
}