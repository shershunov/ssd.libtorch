#include "model.h"
#include <cmath>

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

Net::Net(int64_t num_classes, int64_t num_anchors, float depth_multiple, float width_multiple) : num_classes(num_classes), num_anchors(num_anchors) {
    int64_t p1_out = make_divisible(64 * width_multiple, 8);
    p1 = register_module("p1", std::make_shared<ConvBNSiLU>(3, p1_out, 6, 2, 2));

    int64_t p2_out = make_divisible(128 * width_multiple, 8);
    p2 = register_module("p2", std::make_shared<ConvBNSiLU>(p1_out, p2_out, 3, 2, 1));

    c3_1 = register_module("c3_1", std::make_shared<C3>(p2_out, p2_out, 3, depth_multiple));

    int64_t p3_out = make_divisible(256 * width_multiple, 8);
    p3 = register_module("p3", std::make_shared<ConvBNSiLU>(p2_out, p3_out, 3, 2, 1));

    c3_2 = register_module("c3_2", std::make_shared<C3>(p3_out, p3_out, 6, depth_multiple));

    int64_t p4_out = make_divisible(512 * width_multiple, 8);
    p4 = register_module("p4", std::make_shared<ConvBNSiLU>(p3_out, p4_out, 3, 2, 1));

    c3_3 = register_module("c3_3", std::make_shared<C3>(p4_out, p4_out, 9, depth_multiple));

    int64_t p5_out = make_divisible(1024 * width_multiple, 8);
    p5 = register_module("p5", std::make_shared<ConvBNSiLU>(p4_out, p5_out, 3, 2, 1));

    c3_4 = register_module("c3_4", std::make_shared<C3>(p5_out, p5_out, 3, depth_multiple));

    loc_head_s = register_module("loc_head_s", torch::nn::Conv2d(torch::nn::Conv2dOptions(p3_out, num_anchors * 4, 3).padding(1)));
    conf_head_s = register_module("conf_head_s", torch::nn::Conv2d(torch::nn::Conv2dOptions(p3_out, num_anchors * num_classes, 3).padding(1)));

    loc_head_m = register_module("loc_head_m", torch::nn::Conv2d(torch::nn::Conv2dOptions(p4_out, num_anchors * 4, 3).padding(1)));
    conf_head_m = register_module("conf_head_m", torch::nn::Conv2d(torch::nn::Conv2dOptions(p4_out, num_anchors * num_classes, 3).padding(1)));

    loc_head_l = register_module("loc_head_l", torch::nn::Conv2d(torch::nn::Conv2dOptions(p5_out, num_anchors * 4, 3).padding(1)));
    conf_head_l = register_module("conf_head_l", torch::nn::Conv2d(torch::nn::Conv2dOptions(p5_out, num_anchors * num_classes, 3).padding(1)));

    initialize_weights();
}

std::pair<torch::Tensor, torch::Tensor> Net::forward(torch::Tensor& x) {
    int64_t B = x.size(0);

    x = p1->forward(x);
    x = p2->forward(x);
    x = c3_1->forward(x);

    x = p3->forward(x);
    x = c3_2->forward(x);
    auto feat_s = x;

    x = p4->forward(x);
    x = c3_3->forward(x);
    auto feat_m = x;

    x = p5->forward(x);
    x = c3_4->forward(x);
    auto feat_l = x;

    auto reshape = [B](torch::Tensor x, int64_t d) {
        return x.permute({ 0, 2, 3, 1 }).contiguous().view({ B, -1, d });
    };

    auto loc_s = reshape(loc_head_s->forward(feat_s), 4);
    auto loc_m = reshape(loc_head_m->forward(feat_m), 4);
    auto loc_l = reshape(loc_head_l->forward(feat_l), 4);

    auto conf_s = reshape(conf_head_s->forward(feat_s), num_classes);
    auto conf_m = reshape(conf_head_m->forward(feat_m), num_classes);
    auto conf_l = reshape(conf_head_l->forward(feat_l), num_classes);

    auto loc_preds = torch::cat({ loc_s, loc_m, loc_l }, 1);
    auto conf_preds = torch::cat({ conf_s, conf_m, conf_l }, 1);

    return { loc_preds, conf_preds };
}

void Net::initialize_weights() {
    for (auto& module : modules(false)) {
        if (auto* conv = module->as<torch::nn::Conv2dImpl>()) {
            torch::nn::init::kaiming_normal_(conv->weight, 0.0, torch::kFanOut, torch::kLeakyReLU);
            if (conv->bias.defined()) {
                torch::nn::init::zeros_(conv->bias);
            }
        }
        else if (auto* bn = module->as<torch::nn::BatchNorm2dImpl>()) {
            torch::nn::init::ones_(bn->weight);
            torch::nn::init::zeros_(bn->bias);
        }
    }

    for (auto& head : { conf_head_s, conf_head_m, conf_head_l }) {
        torch::nn::init::normal_(head->weight, 0.0, 0.01);
        torch::nn::init::zeros_(head->bias);
        {
            torch::NoGradGuard no_grad;
            head->bias.view({num_anchors, num_classes}).select(1, 0).fill_(4.6f);
        }
    }

    for (auto& head : { loc_head_s, loc_head_m, loc_head_l }) {
        torch::nn::init::normal_(head->weight, 0.0, 0.01);
        torch::nn::init::zeros_(head->bias);
    }
}
