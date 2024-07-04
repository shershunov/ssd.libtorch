#pragma once
#include <torch/torch.h>
#include <opencv2/opencv.hpp>

torch::Tensor xywh_to_xyxy(const torch::Tensor& xywh);

torch::Tensor iou(const torch::Tensor& boxes1, const torch::Tensor& boxes2);

torch::Tensor normalize_image(cv::Mat& image, const int& img_size);

std::vector<int> non_max_suppression(const torch::Tensor& boxes, const torch::Tensor& scores, float iou_threshold);