#include "model.h"

int64_t make_divisible(int64_t x, int64_t divisor) {
    return (x + divisor / 2) / divisor * divisor;
}

ConvBNSiLU::ConvBNSiLU(int64_t in_channels, int64_t out_channels, int64_t kernel_size, int64_t stride, int64_t padding) {
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding).bias(false)));
    bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
}

torch::Tensor ConvBNSiLU::forward(torch::Tensor& x) {
    return silu(bn->forward(conv->forward(x)));
}

BottleNeck::BottleNeck(int64_t in_channels, int64_t out_channels) {
    conv1 = register_module("conv1", std::make_shared<ConvBNSiLU>(in_channels, out_channels, 1, 1, 0));
    conv2 = register_module("conv2", std::make_shared<ConvBNSiLU>(out_channels, out_channels, 3, 1, 1));
}

torch::Tensor BottleNeck::forward(torch::Tensor x) {
    torch::Tensor y = conv1->forward(x);
    y = conv2->forward(y);
    return y += x;
}

C3::C3(int64_t in_channels, int64_t out_channels, int64_t num_bottlenecks, float depth_multiple) {
    num_bottlenecks = std::max(static_cast<int64_t>(std::round(num_bottlenecks * depth_multiple)), static_cast<int64_t>(1));
    for (int64_t i = 0; i < num_bottlenecks; ++i) {
        bottleneck_layers->push_back(BottleNeck(out_channels / 2, out_channels / 2));
    }
    conv1 = register_module("conv1", std::make_shared<ConvBNSiLU>(in_channels, out_channels / 2, 1, 1, 0));
    conv2 = register_module("conv2", std::make_shared<ConvBNSiLU>(in_channels, out_channels / 2, 1, 1, 0));
    register_module("bottleneck_layers", bottleneck_layers);
    conv3 = register_module("conv3", std::make_shared<ConvBNSiLU>(out_channels, out_channels, 1, 1, 0));
}

torch::Tensor C3::forward(torch::Tensor& x) {
    torch::Tensor y1 = conv1->forward(x);
    y1 = bottleneck_layers->forward(y1);
    torch::Tensor y2 = conv2->forward(x);
    torch::Tensor y = torch::cat({ y1, y2 }, 1);
    return conv3->forward(y);
}

Net::Net(int64_t num_classes, int64_t num_anchors, float depth_multiple, float width_multiple) : num_classes(num_classes), num_anchors(num_anchors)  {
    int64_t p1_out_chanels = make_divisible(64 * width_multiple, 8);
    p1 = register_module("p1", std::make_shared<ConvBNSiLU>(3, p1_out_chanels, 6, 2, 2));

    int64_t p2_out_chanels = make_divisible(128 * width_multiple, 8);
    p2 = register_module("p2", std::make_shared<ConvBNSiLU>(p1_out_chanels, p2_out_chanels, 3, 2, 1));

    c3_1 = register_module("c3_1", std::make_shared<C3>(p2_out_chanels, p2_out_chanels, 3, depth_multiple));

    int64_t p3_out_chanels = make_divisible(256 * width_multiple, 8);
    p3 = register_module("p3", std::make_shared<ConvBNSiLU>(p2_out_chanels, p3_out_chanels, 3, 2, 1));

    c3_2 = register_module("c3_2", std::make_shared<C3>(p3_out_chanels, p3_out_chanels, 6, depth_multiple));

    int64_t p4_out_chanels = make_divisible(512 * width_multiple, 8);
    p4 = register_module("p4", std::make_shared<ConvBNSiLU>(p3_out_chanels, p4_out_chanels, 3, 2, 1));

    c3_3 = register_module("c3_3", std::make_shared<C3>(p4_out_chanels, p4_out_chanels, 9, depth_multiple));

    int64_t p5_out_chanels = make_divisible(1024 * width_multiple, 8);
    p5 = register_module("p5", std::make_shared<ConvBNSiLU>(p4_out_chanels, p5_out_chanels, 3, 2, 1));

    c3_4 = register_module("c3_4", std::make_shared<C3>(p5_out_chanels, p5_out_chanels, 3, depth_multiple));

    int64_t p6_out_chanels = make_divisible(1024 * width_multiple, 8);
    p6 = register_module("p6", std::make_shared<ConvBNSiLU>(p5_out_chanels, p6_out_chanels, 1, 1, 0));

    c3_5 = register_module("c3_5", std::make_shared<C3>(p6_out_chanels, p6_out_chanels, 3, depth_multiple));

    int64_t p7_out_chanels = make_divisible(1024 * width_multiple, 8);
    p7 = register_module("p7", std::make_shared<ConvBNSiLU>(p6_out_chanels, p7_out_chanels, 1, 1, 0));

    c3_6 = register_module("c3_6", std::make_shared<C3>(p7_out_chanels, p7_out_chanels, 3, depth_multiple));

    int64_t p8_out_chanels = make_divisible(1024 * width_multiple, 8);
    p8 = register_module("p8", std::make_shared<ConvBNSiLU>(p7_out_chanels, p8_out_chanels, 1, 1, 0));
    p9 = register_module("p9", std::make_shared<ConvBNSiLU>(p8_out_chanels, p8_out_chanels, 1, 1, 0));
    p10 = register_module("p10", std::make_shared<ConvBNSiLU>(p8_out_chanels, p8_out_chanels, 1, 1, 0));

    loc_layers = register_module("loc_layers", torch::nn::ModuleList());
    conf_layers = register_module("conf_layers", torch::nn::ModuleList());

    for (int i = 0; i < num_anchors; ++i) {
        loc_layers->push_back(register_module("loc_" + std::to_string(i), torch::nn::Conv2d(p8_out_chanels, 4 * num_anchors, 1)));
        conf_layers->push_back(register_module("conf_" + std::to_string(i), torch::nn::Conv2d(p8_out_chanels, num_classes * num_anchors, 1)));
    }
}

std::pair<torch::Tensor, torch::Tensor> Net::forward(torch::Tensor& x) {
    x = p1->forward(x);
    x = p2->forward(x);
    x = c3_1->forward(x);

    x = p3->forward(x);
    x = c3_2->forward(x);

    x = p4->forward(x);
    x = c3_3->forward(x);

    x = p5->forward(x);
    x = c3_4->forward(x);

    x = p6->forward(x);

    x = c3_5->forward(x);

    x = p7->forward(x);

    x = c3_6->forward(x);

    x = p8->forward(x);
    x = p9->forward(x);
    x = p10->forward(x);

    std::vector<torch::Tensor> loc_preds, conf_preds;
    for (int i = 0; i < loc_layers->size(); ++i) {
        loc_preds.push_back((*loc_layers)[i]->as<torch::nn::Conv2d>()->forward(x));
        conf_preds.push_back((*conf_layers)[i]->as<torch::nn::Conv2d>()->forward(x));
    }

    torch::Tensor loc_preds_tensor = torch::cat(loc_preds, 2);
    torch::Tensor conf_preds_tensor = torch::cat(conf_preds, 2);

    loc_preds_tensor = loc_preds_tensor.view({ loc_preds_tensor.size(0), -1, 4 });
    conf_preds_tensor = conf_preds_tensor.view({ conf_preds_tensor.size(0), -1, num_classes });

    return { loc_preds_tensor, conf_preds_tensor };
}
