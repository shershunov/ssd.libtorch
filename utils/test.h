#pragma once
#include <vector>
#include "../model/model.h"
#include "utils.h"


void test_model(Net& model, torch::Device& device, int img_size, std::string dataset_path);

void plot_box(const int x[4], cv::Mat& img, const std::string& label, int line_thickness);

torch::Tensor iou(const torch::Tensor& boxes1, const torch::Tensor& boxes2);