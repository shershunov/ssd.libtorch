#pragma once
#include <torch/torch.h>

struct ConvBNSiLU : torch::nn::Module {
    torch::nn::Conv2d conv{ nullptr };
    torch::nn::BatchNorm2d bn{ nullptr };
    torch::nn::SiLU silu{};

    ConvBNSiLU(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding = 1);

    torch::Tensor forward(torch::Tensor& x);
};

struct BottleNeck : torch::nn::Module {
    std::shared_ptr<ConvBNSiLU> conv1, conv2;

    BottleNeck(int64_t in_channels, int64_t out_channels);

    torch::Tensor forward(torch::Tensor x);
};

struct C3 : torch::nn::Module {
    std::shared_ptr<ConvBNSiLU> conv1, conv2, conv3;
    torch::nn::Sequential bottleneck_layers;

    C3(int64_t in_channels, int64_t out_channels, int64_t num_bottlenecks, float depth_multiple);

    torch::Tensor forward(torch::Tensor& x);
};

class Net : public torch::nn::Module {
public:
    Net(int64_t num_classes, int64_t num_anchors, float depth_multiple, float width_multiple);

    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor& x);

private: 
    int64_t num_classes;
    int64_t num_anchors;
    torch::nn::ModuleList loc_layers, conf_layers;
    torch::nn::MaxPool2d maxpool{ nullptr };
    std::shared_ptr<ConvBNSiLU> p1, p2, p3, p4, p5, p6, p7, p8, p9, p10;
    std::shared_ptr<C3> c3_1, c3_2, c3_3, c3_4, c3_5, c3_6;
};