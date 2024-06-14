#pragma once
#include <torch/torch.h>

struct ConvBNSiLUImpl : torch::nn::Module {
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };
    torch::nn::SiLU silu{};

    ConvBNSiLUImpl(int in_channels, int out_channels, int kernel_size, int stride, int padding = 1);

    torch::Tensor forward(torch::Tensor& x);
};

class Net : public torch::nn::Module {
public:
    Net(const int num_classes, const int num_anchors);

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor& x);

private: 
    int num_classes;
    int num_anchors;
    torch::nn::ModuleList loc_layers, conf_layers;
    torch::nn::MaxPool2d maxpool{ nullptr };
    std::shared_ptr<ConvBNSiLUImpl> conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10,
        conv11, conv12, conv13, conv14, conv15, conv16, conv17, conv18, conv19, conv20, conv21, conv22, conv23, conv24, conv25, conv26, conv27, conv28, conv29, conv30, conv31, 
        conv32, conv33, conv34, conv35, conv36, conv37, conv38, conv39, conv40, extras, loc, conf;
};