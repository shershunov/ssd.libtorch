#pragma once
#include <vector>
#include "model.h"
#include "utils.h"


void test_model(Net& model, torch::Device& device, int width, int height);

cv::Mat plot_box(const int x[4], cv::Mat& img, const std::string& label, int line_thickness, cv::Scalar color);

torch::Tensor iou(const torch::Tensor& boxes1, const torch::Tensor& boxes2);

torch::Tensor xywh_to_xyxy(const torch::Tensor& xywh);