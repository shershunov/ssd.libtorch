#include "utils.h"

torch::Tensor normalize_image(cv::Mat& image, const int& resize_width, const int& resize_height, const torch::Device& device) {
    cv::resize(image, image, cv::Size(resize_width, resize_height));
    image.convertTo(image, CV_32FC3, 1.0 / 255.0);
    torch::Tensor tensor_image = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kFloat32).permute({ 2, 0, 1 });
    tensor_image = torch::data::transforms::Normalize<>({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 })(tensor_image);

    return tensor_image.to(device);
}

torch::Tensor xywh_to_xyxy(const torch::Tensor& xywh) {
    auto xywh_slices = xywh.split(1, 1);
    auto x = xywh_slices[0];
    auto y = xywh_slices[1];
    auto w = xywh_slices[2];
    auto h = xywh_slices[3];

    auto half_w = w / 2;
    auto half_h = h / 2;

    auto x1 = x - half_w;
    auto y1 = y - half_h;
    auto x2 = x + half_w;
    auto y2 = y + half_h;

    return torch::cat({ x1, y1, x2, y2 }, 1);
}

torch::Tensor iou(const torch::Tensor& boxesA, const torch::Tensor& boxesB) {
    auto boxesA_exp = boxesA.unsqueeze(1);
    auto boxesB_exp = boxesB.unsqueeze(0);

    auto xA = torch::max(boxesA_exp.select(2, 0), boxesB_exp.select(2, 0));
    auto yA = torch::max(boxesA_exp.select(2, 1), boxesB_exp.select(2, 1));
    auto xB = torch::min(boxesA_exp.select(2, 2), boxesB_exp.select(2, 2));
    auto yB = torch::min(boxesA_exp.select(2, 3), boxesB_exp.select(2, 3));

    auto interArea = torch::max(xB - xA + 1, torch::zeros_like(xB)) * torch::max(yB - yA + 1, torch::zeros_like(yB));

    auto boxesAArea = (boxesA.select(1, 2) - boxesA.select(1, 0) + 1) * (boxesA.select(1, 3) - boxesA.select(1, 1) + 1);
    auto boxesBArea = (boxesB.select(1, 2) - boxesB.select(1, 0) + 1) * (boxesB.select(1, 3) - boxesB.select(1, 1) + 1);

    auto boxesAArea_exp = boxesAArea.unsqueeze(1);
    auto boxesBArea_exp = boxesBArea.unsqueeze(0);

    auto iou = interArea / (boxesAArea_exp + boxesBArea_exp - interArea);

    return iou;
}