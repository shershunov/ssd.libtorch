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

torch::Tensor iou(const torch::Tensor& boxesA, const torch::Tensor& boxesB) {
    torch::Tensor boxesA_exp = boxesA.unsqueeze(1);
    torch::Tensor boxesB_exp = boxesB.unsqueeze(0);

    torch::Tensor xA = torch::max(boxesA_exp.select(2, 0), boxesB_exp.select(2, 0));
    torch::Tensor yA = torch::max(boxesA_exp.select(2, 1), boxesB_exp.select(2, 1));
    torch::Tensor xB = torch::min(boxesA_exp.select(2, 2), boxesB_exp.select(2, 2));
    torch::Tensor yB = torch::min(boxesA_exp.select(2, 3), boxesB_exp.select(2, 3));

    torch::Tensor interWidth = torch::clamp(xB - xA + 1, 0, std::numeric_limits<float>::infinity());
    torch::Tensor interHeight = torch::clamp(yB - yA + 1, 0, std::numeric_limits<float>::infinity());
    torch::Tensor interArea = interWidth * interHeight;

    torch::Tensor boxesAArea = (boxesA.select(1, 2) - boxesA.select(1, 0) + 1) * (boxesA.select(1, 3) - boxesA.select(1, 1) + 1);
    torch::Tensor boxesBArea = (boxesB.select(1, 2) - boxesB.select(1, 0) + 1) * (boxesB.select(1, 3) - boxesB.select(1, 1) + 1);

    torch::Tensor boxesAArea_exp = boxesAArea.unsqueeze(1);
    torch::Tensor boxesBArea_exp = boxesBArea.unsqueeze(0);

    torch::Tensor unionArea = boxesAArea_exp + boxesBArea_exp - interArea;

    unionArea = torch::max(unionArea, torch::ones_like(unionArea) * 1e-6);

    torch::Tensor iou = interArea / unionArea;

    return iou;
}