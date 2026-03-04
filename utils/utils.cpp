#include "utils.h"

torch::Tensor normalize_image(const cv::Mat& image, const int& img_size) {
    cv::Mat resized, rgb;
    cv::resize(image, resized, cv::Size(img_size, img_size));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
    torch::Tensor tensor_image = torch::from_blob(rgb.data, { rgb.rows, rgb.cols, 3 }, torch::kFloat32);
    tensor_image = tensor_image.permute({ 2, 0, 1 }).clone();
    return tensor_image;
}

torch::Tensor xywh_to_xyxy(const torch::Tensor& xywh) {
    auto slices = xywh.split(1, 1);
    torch::Tensor half_w = slices[2] / 2;
    torch::Tensor half_h = slices[3] / 2;
    return torch::cat({ slices[0] - half_w, slices[1] - half_h, slices[0] + half_w, slices[1] + half_h }, 1);
}

torch::Tensor iou(const torch::Tensor& boxesA, const torch::Tensor& boxesB) {
    torch::Tensor boxesA_exp = boxesA.unsqueeze(1);
    torch::Tensor boxesB_exp = boxesB.unsqueeze(0);

    torch::Tensor xA = torch::max(boxesA_exp.select(2, 0), boxesB_exp.select(2, 0));
    torch::Tensor yA = torch::max(boxesA_exp.select(2, 1), boxesB_exp.select(2, 1));
    torch::Tensor xB = torch::min(boxesA_exp.select(2, 2), boxesB_exp.select(2, 2));
    torch::Tensor yB = torch::min(boxesA_exp.select(2, 3), boxesB_exp.select(2, 3));

    torch::Tensor interArea = torch::clamp_min(xB - xA, 0) * torch::clamp_min(yB - yA, 0);

    torch::Tensor boxesAArea = (boxesA.select(1, 2) - boxesA.select(1, 0)) * (boxesA.select(1, 3) - boxesA.select(1, 1));
    torch::Tensor boxesBArea = (boxesB.select(1, 2) - boxesB.select(1, 0)) * (boxesB.select(1, 3) - boxesB.select(1, 1));

    torch::Tensor unionArea = boxesAArea.unsqueeze(1) + boxesBArea.unsqueeze(0) - interArea;

    return interArea / torch::clamp_min(unionArea, 1e-6f);
}

std::vector<int> non_max_suppression(const torch::Tensor& boxes, const torch::Tensor& scores, float iou_threshold) {
    assert(boxes.size(0) == scores.size(0));

    auto max_scores = std::get<0>(scores.max(1));
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
        auto current_box = boxes[current].unsqueeze(0);

        auto ious = iou(current_box, remaining_boxes).squeeze(0);

        auto mask = ious <= iou_threshold;
        sorted_indices = sorted_indices.masked_select(mask);
    }

    return keep;
}
