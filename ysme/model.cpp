#include "model.h"

ConvBNSiLUImpl::ConvBNSiLUImpl(int in_channels, int out_channels, int kernel_size, int stride, int padding) {
    conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size).stride(stride).padding(padding).bias(false)));
    bn = register_module("bn", torch::nn::BatchNorm2d(out_channels));
}

torch::Tensor ConvBNSiLUImpl::forward(torch::Tensor& x) {
    return silu(bn->forward(conv->forward(x)));
}

Net::Net(int num_classes, int num_anchors) : num_classes(num_classes), num_anchors(num_anchors)  {
    maxpool = register_module("max_pool", torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions({ 2, 2 })));

    conv1 = register_module("conv1", std::make_shared<ConvBNSiLUImpl>(3, 32, 4, 2, 2));
    conv2 = register_module("conv2", std::make_shared<ConvBNSiLUImpl>(32, 64, 2, 2, 1));
    conv3 = register_module("conv3", std::make_shared<ConvBNSiLUImpl>(64, 64, 2, 1, 1));
    conv4 = register_module("conv4", std::make_shared<ConvBNSiLUImpl>(64, 64, 1, 1, 0));
    conv5 = register_module("conv5", std::make_shared<ConvBNSiLUImpl>(64, 64, 1, 1, 0));
    conv6 = register_module("conv6", std::make_shared<ConvBNSiLUImpl>(64, 64, 1, 1, 0));
    conv7 = register_module("conv7", std::make_shared<ConvBNSiLUImpl>(64, 128, 1, 1, 0));
    conv8 = register_module("conv8", std::make_shared<ConvBNSiLUImpl>(128, 128, 1, 1, 0));
    conv9 = register_module("conv9", std::make_shared<ConvBNSiLUImpl>(128, 128, 1, 1, 0));
    conv10 = register_module("conv10", std::make_shared<ConvBNSiLUImpl>(128, 256, 1, 1, 0));
    conv11 = register_module("conv11", std::make_shared<ConvBNSiLUImpl>(256, 256, 1, 1, 0));
    conv12 = register_module("conv12", std::make_shared<ConvBNSiLUImpl>(256, 256, 1, 1, 0));
    conv13 = register_module("conv13", std::make_shared<ConvBNSiLUImpl>(256, 512, 1, 1, 0));
    conv14 = register_module("conv14", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv15 = register_module("conv15", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv16 = register_module("conv16", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv17 = register_module("conv17", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv18 = register_module("conv18", std::make_shared<ConvBNSiLUImpl>(512, 1024, 1, 1, 0));
    conv19 = register_module("conv19", std::make_shared<ConvBNSiLUImpl>(1024, 1024, 1, 1, 0));
    conv20 = register_module("conv20", std::make_shared<ConvBNSiLUImpl>(1024, 1024, 1, 1, 0));
    conv21 = register_module("conv21", std::make_shared<ConvBNSiLUImpl>(1024, 1024, 1, 1, 0));
    conv22 = register_module("conv22", std::make_shared<ConvBNSiLUImpl>(1024, 512, 1, 1, 0));
    conv23 = register_module("conv23", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv24 = register_module("conv24", std::make_shared<ConvBNSiLUImpl>(512, 736, 1, 1, 0));
    conv25 = register_module("conv25", std::make_shared<ConvBNSiLUImpl>(736, 736, 1, 1, 0));
    conv26 = register_module("conv26", std::make_shared<ConvBNSiLUImpl>(736, 512, 1, 1, 0));
    conv27 = register_module("conv27", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv28 = register_module("conv28", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv29 = register_module("conv29", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv30 = register_module("conv30", std::make_shared<ConvBNSiLUImpl>(512, 1024, 1, 1, 0));
    conv31 = register_module("conv31", std::make_shared<ConvBNSiLUImpl>(1024, 1024, 1, 1, 0));
    conv32 = register_module("conv32", std::make_shared<ConvBNSiLUImpl>(1024, 1024, 1, 1, 0));
    conv33 = register_module("conv33", std::make_shared<ConvBNSiLUImpl>(1024, 512, 1, 1, 0));
    conv34 = register_module("conv34", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv35 = register_module("conv35", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv36 = register_module("conv36", std::make_shared<ConvBNSiLUImpl>(512, 512, 1, 1, 0));
    conv37 = register_module("conv37", std::make_shared<ConvBNSiLUImpl>(512, 1024, 1, 1, 0));
    conv38 = register_module("conv38", std::make_shared<ConvBNSiLUImpl>(1024, 1024, 1, 1, 0));
    conv39 = register_module("conv39", std::make_shared<ConvBNSiLUImpl>(1024, 1024, 1, 1, 0));
    conv40 = register_module("conv40", std::make_shared<ConvBNSiLUImpl>(1024, 1024, 1, 1, 0));

    loc_layers = register_module("loc_layers", torch::nn::ModuleList());
    conf_layers = register_module("conf_layers", torch::nn::ModuleList());

    for (int i = 0; i < num_anchors; ++i) {
        loc_layers->push_back(register_module("loc_" + std::to_string(i), torch::nn::Conv2d(1024, 4 * num_anchors, 1)));
        conf_layers->push_back(register_module("conf_" + std::to_string(i), torch::nn::Conv2d(1024, num_classes * num_anchors, 1)));
    }
}

std::pair<torch::Tensor, torch::Tensor> Net::forward(torch::Tensor& x) {
    x = conv1->forward(x);
    x = conv2->forward(x);
    x = maxpool->forward(x);

    x = conv3->forward(x);
    x = conv4->forward(x);
    x = conv5->forward(x);
    x = conv6->forward(x);
    x = maxpool->forward(x);

    x = conv7->forward(x);
    x = conv8->forward(x);
    x = conv9->forward(x);
    x = conv10->forward(x);
    x = maxpool->forward(x);

    x = conv11->forward(x);
    x = conv12->forward(x);
    x = conv13->forward(x);
    x = conv14->forward(x);
    x = conv15->forward(x);
    x = conv16->forward(x);
    x = maxpool->forward(x);

    x = conv17->forward(x);
    x = conv18->forward(x);
    x = conv19->forward(x);
    x = conv20->forward(x);
    x = conv21->forward(x);
    x = conv22->forward(x);
    x = conv23->forward(x);
    x = conv24->forward(x);
    x = conv25->forward(x);
    x = conv26->forward(x);
    x = conv27->forward(x);
    x = conv28->forward(x);
    x = conv29->forward(x);
    x = conv30->forward(x);
    x = conv31->forward(x);

    std::vector<torch::Tensor> loc_preds, conf_preds;
    for (uint16_t i = 0; i < loc_layers->size(); ++i) {
        loc_preds.push_back((*loc_layers)[i]->as<torch::nn::Conv2d>()->forward(x));
        conf_preds.push_back((*conf_layers)[i]->as<torch::nn::Conv2d>()->forward(x));
    }

    torch::Tensor loc_preds_tensor = torch::cat(loc_preds, 2);
    torch::Tensor conf_preds_tensor = torch::cat(conf_preds, 2);

    loc_preds_tensor = loc_preds_tensor.view({ loc_preds_tensor.size(0), -1, 4 });
    conf_preds_tensor = conf_preds_tensor.view({ conf_preds_tensor.size(0), -1, num_classes });

    return { loc_preds_tensor, conf_preds_tensor };
}